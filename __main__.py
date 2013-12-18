#!/usr/bin/env python
import os
from argparse import ArgumentParser
from Tkinter import Tk
from tkFileDialog import askdirectory, askopenfilename

Tk().withdraw()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('rendtype',
                        help='The type of renderer to use, time-dependent or static. Static by default.',
                        choices=['tdi', 'static'])
    parser.add_argument('--simdir',
                        help='Location of the directory containing the simulation.')
    parser.add_argument('--tdiparam',
                        help='Location of the time-dependent ionization parameter file.')
    parser.add_argument('--snap', type=int,
                        help='The number of the snap to display.')
    args = parser.parse_args()

    data_dir = (args.simdir if args.simdir else askdirectory(title='Simulation Directory')).rstrip('/')

    from br_ioni.Renderer import RenderGUI
    if args.rendtype == 'tdi':
        from br_ioni import TDIEmRenderer
        tdi_paramfile_abs = (args.tdiparam if args.tdiparam else
                             askopenfilename(title='Time-dependent Ionization Paramfile'))
        tdi_paramfile = os.path.relpath(tdi_paramfile_abs, data_dir)
        s = TDIEmRenderer(data_dir=data_dir, paramfile=tdi_paramfile, snap=args.snap)
    else:
        from br_ioni import StaticEmRenderer
        s = StaticEmRenderer(data_dir=data_dir, snap=args.snap)
    RenderGUI.show_renderer(s)
