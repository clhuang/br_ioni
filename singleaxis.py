import shlex
import os
import math as m
from collections import namedtuple

import numpy as np
import pycuda.driver as cuda

from bifrost import OSC_data
from bifrost import Rhoeetab
from bifrost import Opatab

BLOCKSIZE = 256
MAXGRIDSIZE = 10000000

EE = 1.602189e-12
HH = 6.626176e-27
CC = 2.99792458e10
CCA = CC * 1e8
HCE = HH / EE * CC * 1e8
HC2 = 2 * HH * CC * 1e24

MP = 1.67262178e-27
KB = 1.3806488e-23

TEMAX = 1e5
RSUN = 695.5  # solar radius in megameters

Egi = namedtuple('Egi', ['ev', 'g', 'label', 'ion'])
Trn = namedtuple('Trn', ['irad', 'jrad', 'alamb', 'a_ul', 'f'])

with open(os.path.dirname(os.path.abspath(__file__)) + '/saioni.cu') as kernelfile:
    SERCUDACODE = kernelfile.read()
with open(os.path.dirname(os.path.abspath(__file__)) + '/saem.cu') as kernelfile:
    TDICUDACODE = kernelfile.read()

DEFAULT_LOC = os.path.expanduser('~') + '/LockheedData/ionismalldata'
DEFAULT_PARAMFILE = 'oxygen-II-VII-iris'


class SingAxisRenderer(object):
    zcutoff = -1.0  # the point at which we are considered to be no longer in the chromosphere
    locph = prev_lambd = float('nan')
    snap = 0

    ux = uy = uz = e = r = oscdata = ka_table = opatab = None

    def __init__(self, cuda_code, xaxis, data_dir=DEFAULT_LOC, snap=None):
        cuda.init()

        from pycuda.tools import make_default_context
        global context
        self.context = make_default_context()

        self.cuda_code = cuda_code

        self.xaxis = xaxis
        self.data_dir = data_dir
        with open(data_dir + '/gpuparam.txt') as gpuparamfile:
            self.template = gpuparamfile.readline().strip() + '%03i'
            self.snap_range = [int(i) for i in gpuparamfile.readline().split()]  # range of timesteps
            if snap is None:
                snap = self.snap_range[0]
            gpuparamfile.readline().strip()
            int(gpuparamfile.readline())
            self.acont_filenames = gpuparamfile.readline().split()

        self.rhoeetab = Rhoeetab(fdir=data_dir)

        self.nrhobin = self.rhoeetab.params['nrhobin']
        self.dmin = m.log(self.rhoeetab.params['rhomin'])
        self.drange = m.log(self.rhoeetab.params['rhomax']) - self.dmin

        self.neibin = self.rhoeetab.params['neibin']
        self.emin = m.log(self.rhoeetab.params['eimin'])
        self.erange = m.log(self.rhoeetab.params['eimax']) - self.emin

        self.set_snap(snap)

    def load_texture(self, name, arr):
        '''
        Loads an array into a texture with a name.

        Address by the name in the kernel code.
        '''
        tex = self.mod.get_texref(name)  # x*y*z
        arr = arr.astype('float32')

        if len(arr.shape) == 3:
            carr = arr.copy('F')
            texarray = numpy3d_to_array(carr, 'F')
            tex.set_array(texarray)
        else:
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, 1)
            tex.set_flags(0)
            cuda.matrix_to_texref(arr, tex, order='F')

        tex.set_address_mode(0, cuda.address_mode.CLAMP)
        tex.set_address_mode(1, cuda.address_mode.CLAMP)
        tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        tex.set_filter_mode(cuda.filter_mode.LINEAR)
        self.textures[name] = tex

    def load_constant(self, name, val):
        '''
        Loads a constant into memory by name in kernel code.

        If val is a float, int, char, etc., it must be wrapped by
        np.float32() or np.int32() or similar.
        '''
        cuda.memcpy_htod(self.mod.get_global(name)[0], val)

    def clear_textures(self):
        self.textures = {}

    def i_rendern(self, channels, axis, reverse, opacity=False):
        '''
        Sum of i_render for multiple channels to get total intensity
        '''
        out = 0

        for channel in channels:
            iout = self.i_render(channel, axis, reverse, opacity=opacity)
            if opacity:
                iout = iout[0]
            out += iout

        return out

    def il_rendern(self, channels, axis, reverse, nout=121, dopp_width_range=1e1, opacity=False):
        '''
        Performs multiple il_renders with different lines and sums them.
        Output is given in terms of wavelength (in angstroms) rather than frequency.
        '''
        lmax = 0
        lmin = float('inf')

        for channel in channels:
            dopp_width = dopp_width_range * self.ny0[channel] /\
                CC * 1e2 * m.sqrt(2 * KB * TEMAX / self.awgt[channel] / MP) / 2
            cmin = CCA / (self.ny0[channel] + dopp_width)
            cmax = CCA / (self.ny0[channel] - dopp_width)
            lmin = min(cmin, lmin)
            lmax = max(cmax, lmax)

        out = np.zeros(self.projection_y_size, self.projection_x_size, nout)
        test_lambdas = np.linspace(lmin, lmax, nout)

        for channel in channels:
            test_freqs = CCA / test_lambdas - self.ny0[channel]
            out += self.il_render(channel, axis, reverse, dnus=test_freqs, opacity=opacity)[0]

        print('Finished rendering channels')

        return (out, test_lambdas)

    def render(self, axis, reverse, output_dims, consts, tables, split_tables,
               spec_render, verbose=True):
        self.clear_textures()
        projection_x_size, projection_y_size = output_dims

        for tup in tables:
            self.load_texture(*tup)
        for tup in consts:
            self.load_constant(*tup)

        if verbose:
            print('Loaded textures, computed emissivities')

        input_shape = self.e.shape

        numsplits = np.ceil(np.product(input_shape) / MAXGRIDSIZE)
        if axis == 'x':
            intaxis = self.xaxis
            intaxis_size = input_shape[0]
            ax_id = 0
        if axis == 'y':
            intaxis = self.yaxis
            intaxis_size = input_shape[1]
            ax_id = 1
        if axis == 'z':
            intaxis = self.zaxis
            intaxis_size = input_shape[2]
            ax_id = 2
        splitsize = np.ceil(intaxis_size / numsplits)

        self.load_constant('projectionXsize', np.int32(projection_x_size))
        self.load_constant('projectionYsize', np.int32(projection_y_size))

        #split_tables is tables to split, table_splits is list of split tables
        table_splits = {name: np.array_split(table, numsplits, ax_id) for name, table in split_tables}
        table_splits['aptex'] = np.array_split(np.gradient(intaxis), numsplits)

        if reverse:
            for name in table_splits:
                table_splits[name].reverse()

        for i in xrange(numsplits):
            start = i * splitsize
            if start + splitsize > intaxis_size:
                splitsize = intaxis_size - start

            if verbose:
                print('Rendering ' + axis + '-coords ' + str(start) + '-' +
                      str(start + splitsize) + ' of ' + str(intaxis_size))

            for name, table_split in table_splits:
                self.load_texture(name, table_split[i])

            data_size = self.projection_x_size * self.projection_y_size
            grid_size = (data_size + BLOCKSIZE - 1) / BLOCKSIZE

            spec_render(self, BLOCKSIZE, grid_size)

    def set_snap(self, snap):
        '''
        Sets the timestamp with which to view the data.
        snap should be an integer between snap_range[0] and snap_range[1], inclusive
        '''
        self.snap = snap
        if snap > self.snap_range[1] or snap < self.snap_range[0]:
            raise ValueError('Time must be in the interval (' + str(self.snap_range[0]) +
                             ', ' + str(self.snap_range[1]) + ')')

        self.oscdata = OSC_data(snap, self.template, fdir=self.data_dir)
        if (self.locph != self.locph):  # axes not loaded yet
            self.update_axes(self.zcutoff)

        self.ux = self.oscdata.getvar('ux')[..., :self.locph]
        self.uy = self.oscdata.getvar('uy')[..., :self.locph]
        self.uz = self.oscdata.getvar('uz')[..., :self.locph]
        self.e = self.oscdata.getvar('e')[..., :self.locph]
        self.r = self.oscdata.getvar('r')[..., :self.locph]


class SAStaticEmRenderer(SingAxisRenderer):
    '''
    Class for rendering emissions of a slice of the sun
    using temperature/density table lookup, assuming static
    ionization equilibrium.
    '''
    # range of electron density, temperature
    enmin = 1e12
    enrange = 1e19
    tgmin = 4.0
    tgrange = 5.04

    # number of bins in the acont lookup table for temperature and electron density
    ntgbin = 505
    nedbin = 71

    def __init__(self, data_dir=DEFAULT_LOC, snap=None):
        '''
        Initializes renderer, and loads data from a directory.
        If snap is none, picks the earliest snap specified by gpuparam.txt.
        '''
        super(SAStaticEmRenderer, self).__init__(SERCUDACODE, data_dir=data_dir)
        self.acont_tables = []
        self.awgt = []
        self.ny0 = []

# lookup tables for temperature and energy
        self.tg_table = self.rhoeetab.get_table('tg')
        self.ne_table = self.rhoeetab.get_table('ne')

        for acontfile in self.acont_filenames:
            mem = np.memmap(data_dir + '/' + acontfile, dtype='float32')
            self.acont_tables.append(mem[:self.ntgbin * self.nedbin])
            self.acont_tables[-1].shape = (self.nedbin, self.ntgbin)
            self.ny0.append(CCA / mem[self.ntgbin * self.nedbin + 1])
            self.awgt.append(mem[self.ntgbin * self.nedbin])
        self.acont_tables = np.array(self.acont_tables)

    def i_render(self, channel, axis, reverse, tau=None, opacity=False, verbose=True, fw=None):
        '''
        Calculates the total intensity of light from a particular POV.

        Channel indicates which emission spectra to look at.
        axis, reverse indicate POV.
        If opacity is True, then looks at tau to see what the
        current opacity is (tau can be left as None to have no initial opacity)

        fw allows setting a different wavelength for opacity calculations
        '''
        tables = [('atex', self.acont_tables[channel]),
                  ('entex', self.ne_table),
                  ('tgtex', self.tg_table)]

        if opacity:
            self.set_lambd(fw)
            tables.append(('katex', self.ka_table))

        consts = [('dmin', np.float32(self.dmin)),
                  ('drange', np.float32(self.drange)),
                  ('emin', np.float32(self.emin)),
                  ('erange', np.float32(self.erange)),
                  ('enmin', np.float32(self.enmin)),
                  ('enrange', np.float32(self.enrange)),
                  ('tgmin', np.float32(self.tgmin)),
                  ('tgrange', np.float32(self.tgrange))]

        split_tables = [('dtex', self.r),
                        ('eetex', self.e)]

        if not opacity or tau is None:
            tau = np.zeros((self.projection_y_size, self.projection_x_size), dtype='float32')
        else:
            if tau.shape != (self.projection_y_size, self.projection_x_size):
                raise Exception('Tau must have shape ' + str(self.projection_y_size) +
                                ' by ' + str(self.projection_x_size))
            tau = tau.astype('float32')

        tempout = np.empty_like(tau)

        def ispec_render(self, blocksize, gridsize):
            frender = self.mod.get_function('iRender')

            # integrates to find the emission
            frender(cuda.Out(tempout), cuda.InOut(tau), np.int8(opacity),
                    block=(blocksize, 1, 1), grid=(gridsize, 1, 1))
            ispec_render.datout += tempout

        ispec_render.datout = np.zeros_like(tempout)

        self.render(axis, reverse, consts, tables, split_tables, ispec_render, verbose)

        if opacity:
            return (ispec_render.datout, tau)
        return ispec_render.datout

    def il_render(self, channel, axis, reverse, nlamb=121, dopp_width_range=1e1,
                  tau=None, opacity=False, dnus=None, verbose=True, fw=None):
        '''
        Calculates intensities as a function of frequency.

        Channel indicates which emission spectra to look at.
        axis, reverse indicate POV.
        nlamb indicates number of frequencies to sample,
        dopp_width_range indicates the range of frequencies to sample
        range is (std. deviation of doppler broadening at 100,000K) * dopp_width_range

        If opacity is True, then looks at tau to see what the
        current opacity is (tau can be left as None to have no initial opacity)

        Returns a tuple of:
            Array of nlamb*ysteps*xsteps, (or some combination of x, y, zsteps) containing intensity data
            Array of nlamb, containing the deviation from nu_0 for each index in the first table
        Uses dnus if specified (list of deviations from the center frequency)
        Otherwise generates test_lambdas using nlamb and dopp_width_range
        dopp_width_range specifies the frequency range (frange = dopp_width_range * dopp_width at tmax)
        '''
        dopp_width0 = self.ny0[channel] / CC * 1e2 * m.sqrt(2 * KB / self.awgt[channel] / MP)

        dny = dopp_width_range * 1.0 / nlamb * dopp_width0 * m.sqrt(TEMAX)
        if dnus is None:
            dnus = np.empty(nlamb, dtype='float32')
            for i in xrange(nlamb):
                dnus[i] = (i - (nlamb - 1) / 2) * dny
        else:
            dnus = np.array(dnus, dtype='float32')

        tables = [('atex', self.acont_tables[channel]),
                  ('entex', self.ne_table),
                  ('tgtex', self.tg_table)]

        if opacity:
            self.set_lambd(fw)
            tables.append(('katex', self.ka_table))

        consts = [('dmin', np.float32(self.dmin)),
                  ('drange', np.float32(self.drange)),
                  ('emin', np.float32(self.emin)),
                  ('erange', np.float32(self.erange)),
                  ('enmin', np.float32(self.enmin)),
                  ('enrange', np.float32(self.enrange)),
                  ('tgmin', np.float32(self.tgmin)),
                  ('tgrange', np.float32(self.tgrange))]

        split_tables = [('dtex', self.r),
                        ('eetex', self.e),
                        ('uxtex', self.ux),
                        ('uytex', self.uy),
                        ('uztex', self.uz)]

        if not opacity or tau is None:
            tau = np.zeros((self.projection_y_size, self.projection_x_size), dtype='float32')
        else:
            if tau.shape != (self.projection_y_size, self.projection_x_size):
                raise Exception('Tau must have shape ' + str(self.projection_y_size) +
                                ' by ' + str(self.projection_x_size))
            tau = tau.astype('float32')

        tempout = np.empty(tau.shape + (dnus.size,), dtype='float32')

        # scope hackery, we're not in python 3
        def ilspec_render(self, blocksize, gridsize):
            frender = self.mod.get_function('ilRender')
            frender(cuda.Out(tempout), cuda.In(dnus), cuda.InOut(tau),
                    np.float32(self.ny0[channel]), np.float32(dopp_width0),
                    np.int32(dnus.size), np.int8(opacity),
                    block=(blocksize, 1, 1), grid=(gridsize, 1, 1))

            ilspec_render.datout += tempout

        ilspec_render.datout = np.zeros_like(tempout)

        self.render(axis, reverse, consts, tables, split_tables, ilspec_render, verbose)

        if opacity:
            return (ilspec_render.datout, dnus, tau)
        return (ilspec_render.datout, dnus)

    def channellist(self):
        return self.acont_filenames

    def set_lambd(self, lambd):
        if self.opatab is None:
            self.opatab = Opatab(fdir=self.data_dir)
        if lambd is None:
            lambd = CCA / self.ny0[0]
        if lambd != self.prev_lambd:
            self.prev_lambd = lambd
            self.ka_table = self.opatab.h_he_absorb(lambd)


class SATDIEmRenderer(SingAxisRenderer):
    '''
    Class for rendering emissions of a slice of the sun
    using calculated ion densities.
    '''
    #cache emissivity for speed when rendering
    level = -1
    em = None

    def __init__(self, data_dir=DEFAULT_LOC, paramfile='oxygen-II-VII-iris', snap=None):
        '''
        Initializes renderer, and loads data from a directory and paramfile.
        If snap is none, picks the earliest snap specified by gpuparam.txt.
        '''
        super(SATDIEmRenderer, self).__init__(TDICUDACODE, data_dir=data_dir)

        rhoeetab = Rhoeetab(fdir=data_dir)

        self.nrhobin = rhoeetab.params['nrhobin']
        self.dmin = m.log(rhoeetab.params['rhomin'])
        self.drange = m.log(rhoeetab.params['rhomax']) - self.dmin
        self.tg_table = rhoeetab.get_table('tg')

        self.neibin = rhoeetab.params['neibin']
        self.emin = m.log(rhoeetab.params['eimin'])
        self.erange = m.log(rhoeetab.params['eimax']) - self.emin

        irisfile = open(data_dir + '/' + paramfile)
        data = [line for line in irisfile.readlines() if line[0] != '*']

        self.element_name = data.pop(0).strip()
        self.ab, self.awgt = (float(i) for i in data.pop(0).split())
        nk, nlines, _, _ = (int(i) for i in data.pop(0).split())
        self.egis = []  # array of (ev, g, ion)
        self.trns = []

        for _ in xrange(nk):
            datstring = shlex.split(data.pop(0))
            ev = float(datstring[0]) * CC * HH / EE
            g = float(datstring[1])
            label = datstring[2]
            ion = int(datstring[3])
            self.egis.append(Egi(ev, g, label, ion))

        for _ in xrange(nlines):
            j, i, f, _, _, _, _, _, _, _ = (float(i) for i in data.pop(0).split())
            j = int(j)
            i = int(i)

            dn, up = i, j
            if self.egis[j - 1].ev < self.egis[i - 1].ev:
                dn, up = j, i
            irad = dn - 1
            jrad = up - 1

            alamb = HCE / (self.egis[jrad].ev - self.egis[irad].ev)
            a_ul = f * 6.6702e15 * self.egis[irad].g / (self.egis[jrad].g * alamb ** 2)

            self.trns.append(Trn(irad, jrad, alamb, a_ul, f))

    def i_render(self, level, axis, reverse, tau=None, opacity=False, verbose=True, fw=None):
        '''
        Calculates the total intensity of light from a particular POV.

        Level indicates what emission spectra to look at.
        axis, reverse indicate POV.
        If opacity is True, then looks at tau to see what the
        current opacity is (tau can be left as None to have no initial opacity)

        fw allows setting a different wavelength for opacity calculations
        '''
        if level != self.level:
            self.em = self.get_emissivities(level)
            self.level = level
        axis = ord(axis.lower()) - ord('x')

        consts = []
        tables = []
        split_tables = [('emtex', self.em)]

        if opacity:
            self.set_lambd(fw)
            tables.append(('katex', self.ka_table))
            split_tables.extend([('dtex', self.r),
                                ('eetex', self.e)])
            consts.extend([('dmin', np.float32(self.dmin)),
                           ('drange', np.float32(self.drange)),
                           ('emin', np.float32(self.emin)),
                           ('erange', np.float32(self.erange)),
                           ('axis', np.int8(ord(axis.lower()) - ord('x')))])

        if not opacity or tau is None:
            tau = np.zeros((self.projection_y_size, self.projection_x_size), dtype='float32')
        else:
            if tau.shape != (self.projection_y_size, self.projection_x_size):
                raise Exception('Tau must have shape ' + str(self.projection_y_size) +
                                ' by ' + str(self.projection_x_size))
            tau = tau.astype('float32')

        tempout = np.empty_like(tau)

        def ispec_render(self, blocksize, gridsize):
            frender = self.mod.get_function('iRender')
            frender(cuda.Out(tempout), cuda.InOut(tau), np.int8(opacity),
                    block=(blocksize, 1, 1), grid=(gridsize, 1, 1))
            ispec_render.datout += tempout

        ispec_render.datout = np.zeros_like(tempout)

        self.render(0, 0, consts, tables, split_tables, ispec_render, verbose)
        if opacity:
            return (ispec_render.datout, tau)
        return ispec_render.datout

    def il_render(self, level, axis, reverse, nlamb=121, dopp_width_range=1e1,
                  tau=None, opacity=False, dnus=None, verbose=True, fw=None):
        '''
        Calculates intensities as a function of frequency.

        Level indicates what emission spectra to look at.
        axis, reverse indicate POV.
        nlamb indicates number of frequencies to sample,
        dopp_width_range indicates the range of frequencies to sample
        range is (std. deviation of doppler broadening at 100,000K) * dopp_width_range

        If opacity is True, then looks at tau to see what the
        current opacity is (tau can be left as None to have no initial opacity)

        Returns a tuple of:
            Array of nlamb*ysteps*xsteps, (or some combination of x, y, zsteps) containing intensity data
            Array of nlamb, containing the deviation from nu_0 for each index in the first table
        Uses dnus if specified (list of deviations from the center frequency)
        Otherwise generates test_lambdas using nlamb and dopp_width_range
        dopp_width_range specifies the frequency range (frange = dopp_width_range * dopp_width at tmax)
        '''
        if level != self.level:
            self.em = self.get_emissivities(level)
            self.level = level

        ny0 = CCA / self.trns[level].alamb
        dopp_width0 = ny0 / (CC / 1e2) * m.sqrt(2 * KB / self.awgt / MP)

        dny = dopp_width_range * 1.0 / nlamb * dopp_width0 * m.sqrt(TEMAX)
        if dnus is None:
            dnus = np.empty(nlamb, dtype='float32')
            for i in xrange(nlamb):
                dnus[i] = (i - (nlamb - 1) / 2) * dny
        else:
            dnus = np.array(dnus, dtype='float32')

        consts = [('dmin', np.float32(self.dmin)),
                  ('drange', np.float32(self.drange)),
                  ('emin', np.float32(self.emin)),
                  ('erange', np.float32(self.erange))]
        tables = [('tgtex', self.tg_table)]
        split_tables = [('emtex', self.em),
                        ('uztex', self.uz),
                        ('uytex', self.uy),
                        ('uxtex', self.ux),
                        ('eetex', self.e),
                        ('dtex', self.r)]

        if opacity:
            self.set_lambd(fw)
            tables.append(('katex', self.ka_table))

        if not opacity or tau is None:
            tau = np.zeros((self.projection_y_size, self.projection_x_size), dtype='float32')
        else:
            if tau.shape != (self.projection_y_size, self.projection_x_size):
                raise Exception('Tau must have shape ' + str(self.projection_y_size) +
                                ' by ' + str(self.projection_x_size))
            tau = tau.astype('float32')

        tempout = np.empty(tau.shape + (dnus.size,), dtype='float32')

        def ilspec_render(self, blocksize, gridsize):
            frender = self.mod.get_function('ilRender')
            frender(cuda.Out(tempout), cuda.In(dnus), cuda.InOut(tau),
                    np.float32(ny0), np.float32(dopp_width0),
                    np.int32(nlamb), np.int8(opacity),
                    block=(blocksize, 1, 1), grid=(gridsize, 1, 1))
            ilspec_render.datout += tempout

        ilspec_render.datout = np.zeros_like(tempout)

        self.render(axis, reverse, consts, tables, split_tables, ilspec_render, verbose)
        if opacity:
            return (ilspec_render.datout, dnus, tau)
        return (ilspec_render.datout, dnus)

    def get_emissivities(self, level=None):
        '''
        Returns a 3d table giving the emissivity at each point.
        If level is None, prompts will guide you through selecting
        a level.
        '''
        if (level is None):
            tion = int(input('Input the oxygen ionization level to check: '))
            ionln = [i for i, v in enumerate(self.egis) if v.ion == tion]

            print('These are the ' + self.element_name + str(tion) + ' lines:')
            print('nr\tlvl l\tlvl u\tname lvl l\tname lvl u\twavelength\tEinstein Aij\tion')
            print('--------------------------------------------------------------------------------')

            for n, t in enumerate(self.trns):
                for ind in ionln:
                    if t.jrad == ind or t.irad == ind:
                        print(str(n) + ')\t' + str(t.irad + 1) + '\t' + str(t.jrad + 1) + '\t' +
                              self.egis[t.irad].label + '\t' + self.egis[t.jrad].label + '\t' +
                              '%e' % t.alamb + '\t' + '%e' % t.a_ul + '\t' + str(self.egis[t.irad].ion))
                        break

            level = int(input('Choose a line: '))

        tline = self.trns[level]
        ion_densities = self.oscdata.getooevar(tline.jrad)
        return ((HH * CC * 1e-2 / (tline.alamb * 1e-10) * tline.a_ul / (4 * m.pi)) * ion_densities)[..., :self.locph]

    def channellist(self):
        return [self.egis[t.irad].label + " -->" + self.egis[t.jrad].label for t in self.trns]

    def set_lambd(self, lambd):
        if self.opatab is None:
            self.opatab = Opatab(fdir=self.data_dir)
        if lambd is None:
            lambd = self.trns[self.level].alamb
        if lambd != self.prev_lambd:
            self.prev_lambd = lambd
            self.ka_table = self.opatab.h_he_absorb(lambd)


def numpy3d_to_array(np_array, order=None):
    '''
    Method for copying a numpy array to a CUDA array

    If you get a buffer error, run this method on np_array.copy('F')
    '''
    from pycuda.driver import Array, ArrayDescriptor3D, Memcpy3D, dtype_to_array_format
    if order is None:
        order = 'C' if np_array.strides[0] > np_array.strides[2] else 'F'

    if order.upper() == 'C':
        d, h, w = np_array.shape
    elif order.upper() == "F":
        w, h, d = np_array.shape
    else:
        raise Exception("order must be either F or C")

    descr = ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    device_array = Array(descr)

    copy = Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d

    copy()

    return device_array
