import shlex
import os
import math as m
from collections import namedtuple

import numpy as np
import pycuda.driver as cuda

from bifrost import OSC_data
from bifrost import Rhoeetab
from bifrost import Opatab

from Renderer import SingAxisRenderer

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


class SAEmissivityRenderer(SingAxisRenderer):
    def __init__(self, cuda_code, data_dir=DEFAULT_LOC, snap=None):
        super(SAEmissivityRenderer, self).__init__(cuda_code)

        self.data_dir = data_dir
        with open(data_dir + '/gpuparam.txt') as gpuparamfile:
            self.template = gpuparamfile.readline().strip() + '%03i'
            self.snap_range = [int(i) for i in gpuparamfile.readline().split()]  # range of timesteps
            if snap is None:
                snap = self.snap_range[0]
            gpuparamfile.readline().strip()
            int(gpuparamfile.readline())
            self.acont_filenames = ['data/' + i for i in gpuparamfile.readline().split()]

        self.rhoeetab = Rhoeetab(fdir=data_dir)

        self.nrhobin = self.rhoeetab.params['nrhobin']
        self.dmin = m.log(self.rhoeetab.params['rhomin'])
        self.drange = m.log(self.rhoeetab.params['rhomax']) - self.dmin

        self.neibin = self.rhoeetab.params['neibin']
        self.emin = m.log(self.rhoeetab.params['eimin'])
        self.erange = m.log(self.rhoeetab.params['eimax']) - self.emin

        self.set_snap(snap)

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
            self.xaxis = self.oscdata.getvar('x').astype('float32')
            self.yaxis = self.oscdata.getvar('y').astype('float32')
            self.zaxis = self.oscdata.getvar('z').astype('float32')
            for i, val in enumerate(self.zaxis):
                if val > self.zcutoff:
                    self.locph = i
                    break

        self.ux = self.oscdata.getvar('ux')[..., :self.locph]
        self.uy = self.oscdata.getvar('uy')[..., :self.locph]
        self.uz = self.oscdata.getvar('uz')[..., :self.locph]
        self.e = self.oscdata.getvar('e')[..., :self.locph]
        self.r = self.oscdata.getvar('r')[..., :self.locph]

    def i_render(self, channel, azimuth, altitude, tau=None, opacity=False, verbose=True, fw=None):
        '''
        Calculates the total intensity of light from a particular POV.

        Channel indicates which emission spectra to look at.
        Azimuth, altitude indicate POV.
        If opacity is True, then looks at tau to see what the
        current opacity is (tau can be left as None to have no initial opacity)

        fw allows setting a different wavelength for opacity calculations
        '''

    def il_render(self, channel, azimuth, altitude, nlamb=121, dopp_width_range=1e1,
                  tau=None, opacity=False, dnus=None, verbose=True, fw=None):
        '''
        Calculates intensities as a function of frequency.

        Channel indicates which emission spectra to look at.
        Azimuth, altitude indicate POV.
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


class SAStaticEmRenderer(SAEmissivityRenderer):
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

    def i_render(self, channel, axis, reverse=False, tau=None, opacity=False, verbose=True, fw=None):
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
                        ('uatex', self.ux if axis == 'x' else self.uy if axis == 'y' else self.uz)]

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

    def set_lambd(self, lambd):
        if self.opatab is None:
            self.opatab = Opatab(fdir=self.data_dir)
        if lambd is None:
            lambd = CCA / self.ny0[0]
        if lambd != self.prev_lambd:
            self.prev_lambd = lambd
            self.ka_table = self.opatab.h_he_absorb(lambd)


class SATDIEmRenderer(SAEmissivityRenderer):
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
                           ('erange', np.float32(self.erange))])

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
                        ('uatex', self.ux if axis == 'x' else self.uy if axis == 'y' else self.uz),
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

    def set_lambd(self, lambd):
        if self.opatab is None:
            self.opatab = Opatab(fdir=self.data_dir)
        if lambd is None:
            lambd = self.trns[self.level].alamb
        if lambd != self.prev_lambd:
            self.prev_lambd = lambd
            self.ka_table = self.opatab.h_he_absorb(lambd)
