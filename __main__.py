#!/usr/bin/env python
import os
import glob
from threading import Thread
from argparse import ArgumentParser
from Tkinter import Tk
from tkFileDialog import askopenfilename
from Renderer.renderercontroller import RendererController

Tk().withdraw()

rend = None


def renderer_process_init(rendertype, *args, **kwargs):
    global rend

    if rendertype == 'tdi':
        from br_ioni import TDIEmRenderer as Renderclass
    else:
        from br_ioni import StaticEmRenderer as Renderclass

    rend = Renderclass(*args, **kwargs)
    rend.controller = RendererController()
    rend.controller.start(rend)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('rendtype',
                        help='The type of renderer to use, time-dependent or static. Static by default.',
                        choices=['tdi', 'static'])
    parser.add_argument('--tdiparam',
                        help='Location of the time-dependent ionization parameter file.')
    parser.add_argument('--infile',
                        help='Location of the first simulation file to use.')
    args = parser.parse_args()

#get initial file
    initial_file = (args.infile if args.infile else
                    askopenfilename(title='Choose Initial Snap',
                                    filetypes=[('Aux files', '.aux')]))
    data_dir = os.path.dirname(initial_file)
    initial_file = os.path.basename(initial_file)
    initial_file = os.path.splitext(initial_file)[0]

#determine template format
    template = initial_file.rstrip('0123456789')
    snapstr = initial_file[len(template):]
    template += '%%0%ii' % len(snapstr)  # ends up as somthing like qsmag_by01_it%03i
    snap = int(snapstr)

#determine snap bounds
    snap_min = snap_max = snap
    while (os.path.exists(data_dir + '/' + template % (snap_min - 1) + '.aux')):
        snap_min -= 1
    while (os.path.exists(data_dir + '/' + template % (snap_max + 1) + '.aux')):
        snap_max += 1

    snap_range = (snap_min, snap_max)

    acont_filenames = [os.path.relpath(i, data_dir) for i in glob.glob(data_dir + '/data/*.dat')]

#and render
    from br_ioni.Renderer import RenderGUI
    if args.rendtype == 'tdi':
#prompt for time-dependent parameter file if necessary
        tdi_paramfile_abs = (args.tdiparam if args.tdiparam else
                             askopenfilename(title='Time-dependent Ionization Paramfile'))
        tdi_paramfile = os.path.relpath(tdi_paramfile_abs, data_dir)

        t = Thread(target=renderer_process_init, args=('tdi'),
                   kwargs=dict(data_dir=data_dir, paramfile=tdi_paramfile, snap=snap))
    else:
        t = Thread(target=renderer_process_init, args=('static', snap_range, acont_filenames, template),
                   kwargs=dict(data_dir=data_dir, snap=snap))
    t.start()

    while rend is None:
        pass

    RenderGUI.show_renderer(rend)
