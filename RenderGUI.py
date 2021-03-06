from __future__ import print_function

import os.path
import numpy as np
from string import Template
from br_ioni import spectAnlys, CC, noisify_spectra

import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.properties import NumericProperty, BooleanProperty
from kivy.config import ConfigParser
from kivy.uix.settings import SettingsPanel, SettingOptions, SettingNumeric, SettingBoolean
from kivy.uix.popup import Popup
from Tkinter import Tk
from matplotlib import cm, colors
import tkFileDialog

kivy.require('1.8.0')

Tk().withdraw()


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(map(reduced_cmap, step_list))
    new_LUT = np.array(map(function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return colors.LinearSegmentedColormap('colormap', cdict, 1024)

BUF_DIMENSIONS = (3840, 2160)  # supports up to 4k screens
LOGNORM = colors.LogNorm()
SCALAR_MAP = cm.ScalarMappable(LOGNORM, cm.bone)
SYM_MAP = cmap_map(lambda x: 1 - x, cm.RdBu)


class Mode(object):
    intensity = 'Intensity'
    doppler_shift = 'Doppler Shift'
    width = 'FWHM'
    asym = 'Asymmetries'


class RenderGUI(Widget):
    rend = None
    azimuth = NumericProperty(20.0)
    altitude = NumericProperty(20.0)
    distance_per_pixel = NumericProperty(0.0)
    stepsize = NumericProperty(0.0)
    x_pixel_offset = NumericProperty(0)
    y_pixel_offset = NumericProperty(0)
    rend_opacity = BooleanProperty(False)
    channel = NumericProperty(0)
    log_offset = NumericProperty(6.0)
    cbar_num = NumericProperty(10)
    snap = NumericProperty(0)
    rendermode = Mode.intensity
    spect_analyzer = spectAnlys.Analyzer()
    nlamb = NumericProperty(41)
    cbsize = (30, 3000)
    asym_sep = NumericProperty(0.0)
    asym_width = NumericProperty(0.0)
    noise_snr = NumericProperty(999.)

    helptext = ('Pan l/r: a/d\n'
                'Tilt u/d: w/s\n'
                'zoom in/out: j/k\n'
                'Shift l/r: [left]/[right]\n'
                'Shift u/d: [up]/[down]\n'
                'Recenter shift: c\n'
                'Dynamic range inc/dec: i/u\n'
                'Stepsize inc/dec: ./,\n'
                'Toggle opacity: o\n'
                'Change timestep: [/]')
    initialized = False

    def __init__(self, rend, **kwargs):
        self.texture = Texture.create(size=BUF_DIMENSIONS)
        self.texture_size = BUF_DIMENSIONS
        self.cbtex = Texture.create(size=self.cbsize)
        super(RenderGUI, self).__init__(**kwargs)

        self.rend = rend
        self.buffer_array = np.empty(BUF_DIMENSIONS[::-1] + (4, ), dtype='uint8')
        self.distance_per_pixel = self.rend.distance_per_pixel
        self.stepsize = self.rend.stepsize

        self.x_pixel_offset = rend.x_pixel_offset
        self.y_pixel_offset = rend.y_pixel_offset
        self.snap = self.rend.snap

        self.config = ConfigParser()
        self.channellist = [os.path.basename(os.path.splitext(a)[0]) for a in self.rend.channellist()]
        self.config.setdefaults('renderer', {'rendermode': self.rendermode,
                                             'channel': self.channellist[0],
                                             'snap': self.rend.snap,
                                             'nlamb': self.nlamb,
                                             'opacity': int(self.rend_opacity),
                                             'altitude': self.altitude,
                                             'azimuth': self.azimuth,
                                             'distance_per_pixel': self.distance_per_pixel,
                                             'stepsize': self.stepsize,
                                             'noise_snr': self.noise_snr})
        self.config.setdefaults('display', {'log_offset': self.log_offset,
                                            'cbar_num': self.cbar_num,
                                            'asym_sep': self.asym_sep,
                                            'asym_width': self.asym_width})

        self.spanel = SettingsPanel(settings=self.s, title='Render Settings', config=self.config)
        self.s.interface.add_panel(self.spanel, 'Render Settings', self.spanel.uid)

        self.dpanel = SettingsPanel(settings=self.s, title='Display Settings', config=self.config)
        self.s.interface.add_panel(self.dpanel, 'Display Settings', self.dpanel.uid)

        self.mode_opt = SettingOptions(title='Render Mode',
                                       desc='What to simulate and display',
                                       key='rendermode',
                                       section='renderer',
                                       options=[Mode.__dict__[x] for x in dir(Mode) if not x.startswith('_')],
                                       panel=self.spanel)
        self.spanel.add_widget(self.mode_opt)

        self.chan_opt = SettingOptions(title='Channel',
                                       desc='Emissions channel to select',
                                       key='channel',
                                       section='renderer',
                                       options=self.channellist,
                                       panel=self.spanel)
        self.spanel.add_widget(self.chan_opt)

        self.snap_opt = SettingNumeric(title='Snap',
                                       desc='Snap number to select',
                                       key='snap',
                                       section='renderer',
                                       panel=self.spanel)
        self.spanel.add_widget(self.snap_opt)

        self.nlamb_opt = SettingNumeric(title='NLamb',
                                        desc='Number of frequencies to sample during spectra calculations',
                                        key='nlamb',
                                        section='renderer',
                                        panel=self.spanel)
        self.spanel.add_widget(self.nlamb_opt)

        self.opa_opt = SettingBoolean(title='Opacity',
                                      desc='Whether or not to enable opacity in the simulation',
                                      key='opacity',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.opa_opt)

        self.alt_opt = SettingNumeric(title='Altitude',
                                      desc='The POV angle above horizontal',
                                      key='altitude',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.alt_opt)

        self.azi_opt = SettingNumeric(title='Azimuth',
                                      desc='The POV angle lateral to the x-axis',
                                      key='azimuth',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.azi_opt)

        self.dpp_opt = SettingNumeric(title='Distance per Pixel',
                                      desc='Distance in simulation between pixels in km, specifies zoom',
                                      key='distance_per_pixel',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.dpp_opt)

        self.stp_opt = SettingNumeric(title='Step Size',
                                      desc='Magnitude of the integration stepsize, increase for performance',
                                      key='stepsize',
                                      section='renderer',
                                      panel=self.spanel)
        self.spanel.add_widget(self.stp_opt)

        self.noise_snr_opt = SettingNumeric(title='Spectral SNR',
                                            desc=u'Spectral signal to noise ratio, in dB\u2014to disable, set to 999',
                                            key='noise_snr',
                                            section='renderer',
                                            panel=self.spanel)
        self.spanel.add_widget(self.noise_snr_opt)

        self.range_opt = SettingNumeric(title='Dynamic Range',
                                        desc='Orders of magnitude to span in display',
                                        key='log_offset',
                                        section='display',
                                        panel=self.spanel)
        self.dpanel.add_widget(self.range_opt)

        self.cbarnum_opt = SettingNumeric(title='Colorbar Numbers',
                                          desc='Number of data points to indicate on the colorbar',
                                          key='cbar_num',
                                          section='display',
                                          panel=self.spanel)
        self.dpanel.add_widget(self.cbarnum_opt)

        self.asym_width_opt = SettingNumeric(title='Asymmetry Window Width',
                                             desc='Width of integration window, in km/s',
                                             key='asym_width',
                                             section='display',
                                             panel=self.spanel)
        self.dpanel.add_widget(self.asym_width_opt)

        self.asym_sep_opt = SettingNumeric(title='Asymmetry Window Separation',
                                           desc='Separation of integration windows, in km/s',
                                           key='asym_sep',
                                           section='display',
                                           panel=self.spanel)
        self.dpanel.add_widget(self.asym_sep_opt)

        self._keyboard_open()
        Window.bind(on_resize=self._on_resize)

        #initial update
        self._on_resize(Window, Window.size[0], Window.size[1])
        self._saverangedialog = SaveRangeDialog(self, size_hint=(.8, .8), title="Save Range")

        self.initialized = True

    def _settings_change(self, section, key, value):
        '''
        Called on setting panel change, updates values in renderer config
        '''
        self._keyboard_open()
        if key == 'opacity':
            self.rend_opacity = (value == '1')
        elif key in ('snap', 'nlamb'):
            setattr(self, key, int(value))
        elif key == 'channel':
            self.channel = self.channellist.index(value)
        elif key == 'rendermode':
            self.rendermode = value
        elif key in ('rendermode', 'altitude', 'azimuth', 'distance_per_pixel', 'stepsize',
                     'log_offset', 'cbar_num', 'asym_width', 'asym_sep', 'noise_snr'):
            setattr(self, key, float(value))
        else:
            return
        if section == 'renderer':
            self.update()
        else:
            self.update_display()

    def _keyboard_open(self):
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        '''
        Does stuff on some keypresses
        '''
        if keycode[1] == 'w':  # view up
            self.altitude += 2
        elif keycode[1] == 's':  # view down
            self.altitude -= 2
        elif keycode[1] == 'a':  # view left
            self.azimuth -= 2
        elif keycode[1] == 'd':  # view right
            self.azimuth += 2
        elif keycode[1] == 'j':  # zoom in
            self.distance_per_pixel *= 0.95
        elif keycode[1] == 'k':  # zoom out
            self.distance_per_pixel /= 0.95
        elif keycode[1] == 'u':  # decrease contrast, increasing dyn range
            self.log_offset += 0.4
            self.update_display()  # don't rerender, just update display
            return
        elif keycode[1] == 'i':  # increase contrast
            if self.log_offset > 0:
                self.log_offset -= 0.4
            self.update_display()
            return
        elif keycode[1] == 'up':  # shift view up
            self.y_pixel_offset += 5
        elif keycode[1] == 'down':  # shift view down
            self.y_pixel_offset -= 5
        elif keycode[1] == 'left':  # shift view left
            self.x_pixel_offset -= 5
        elif keycode[1] == 'right':  # shift view right
            self.x_pixel_offset += 5
        elif keycode[1] == 'c':
            self.x_pixel_offset = self.y_pixel_offset = 0
        elif keycode[1] == ',':  # decreases stepsize, increasing resolution
            self.stepsize *= 0.8
        elif keycode[1] == '.':  # increases stepsize, decreasing resolution
            self.stepsize /= 0.8
        elif keycode[1] == '[':  # go back 1 snap
            self.snap -= 1
        elif keycode[1] == ']':  # go forward 1 snap
            self.snap += 1
        elif keycode[1] == 'o':  # toggle opacity
            self.rend_opacity = not self.rend_opacity
        else:
            return

        self.alt_opt.value = str(self.altitude)
        self.azi_opt.value = str(self.azimuth)
        self.range_opt.value = str(self.log_offset)
        self.dpp_opt.value = str(round(self.distance_per_pixel, 6))
        self.stp_opt.value = str(round(self.stepsize, 6))
        self.opa_opt.value = '1' if self.rend_opacity else '0'
        self.snap_opt.value = str(self.rend.snap)
        self.update()

    def _on_resize(self, window, width, height):
        '''
        Rerenders, resizes objects on window resize
        '''
        self.rend.projection_x_size, self.rend.projection_y_size = width, height
        self.s.size = (self.s.size[0], height)
        self.cbsize = (self.cbsize[0], height - 100)
        self.update()

    def update(self, updatedisplay=True):
        '''
        Rerenders stuff and caches it, then updates display if specified
        '''
        if not self.initialized:
            return
        # limit some values
        self.azimuth = self.azimuth % 360
        self.altitude = sorted((-90, self.altitude, 90))[1]
        self.snap = sorted(self.rend.snap_range + (self.snap,))[1]

        # set values in renderer, and render
        self.rend.distance_per_pixel = self.distance_per_pixel
        self.rend.stepsize = self.stepsize
        self.rend.y_pixel_offset = self.y_pixel_offset
        self.rend.x_pixel_offset = self.x_pixel_offset
        self.rend.set_snap(self.snap)

        # render appropriate data, cache it
        if self.rendermode == Mode.intensity:
            data, _ = self.get_i_render()
            self.raw_spectra = None
            self.raw_data = data
        else:
            data, dfreqs, ny0, _ = self.get_il_render()
            self.raw_spectra = (noisify_spectra(data, self.noise_snr), dfreqs, ny0)
            self.raw_data = None
            self.spect_analyzer.set_data(*self.raw_spectra)

        if updatedisplay:
            self.update_display()

    def update_display(self):
        '''
        Rejiggers display objects if no rerendering is required
        '''
        if self.rendermode == Mode.intensity:
            self.unittxt.text = 'Intensity: erg s[sup]-1[/sup] cm[sup]-2[/sup] sr[sup]-1[/sup]'
        elif self.rendermode == Mode.doppler_shift:
            self.raw_data = self.spect_analyzer.quad_regc()
            self.raw_data *= -CC / 1e3 / self.spect_analyzer.center_freq  # convert to km/s
            self.unittxt.text = 'Doppler shift: km/s'
        elif self.rendermode == Mode.width:
            self.raw_data = self.spect_analyzer.fwhm()
            self.raw_data *= CC / 1e3 / self.spect_analyzer.center_freq  # convert to km/s
            self.unittxt.text = 'Line width at half max: km/s'
        elif self.rendermode == Mode.asym:
            self.raw_data = self.spect_analyzer.split_integral_vel(self.asym_sep, self.asym_width, 2)
            self.raw_data = self.raw_data[..., 1] - self.raw_data[..., 0]
            self.unittxt.text = 'Intensity: erg s[sup]-1[/sup] cm[sup]-2[/sup] sr[sup]-1[/sup]'

        bounds = (np.nanmin(self.raw_data), np.nanmax(self.raw_data))
        if bounds[0] >= 0:  # use symlog-based approach starting from 0
            SCALAR_MAP.set_norm(colors.SymLogNorm(bounds[1] * 0.1 ** self.log_offset))
            SCALAR_MAP.set_cmap(cm.bone)
            SCALAR_MAP.set_clim(0, bounds[1])
        else:  # use symlog approach
            b2 = max((abs(bounds[0]), bounds[1]))
            SCALAR_MAP.set_cmap(SYM_MAP)
            SCALAR_MAP.set_norm(colors.SymLogNorm(b2 * 0.1 ** self.log_offset))
            SCALAR_MAP.set_clim(-b2, b2)

        data = SCALAR_MAP.to_rgba(self.raw_data) * 255

        # update display buffer
        self.buffer_array[:data.shape[0], :data.shape[1]] = data
        self.texture.blit_buffer(self.buffer_array.tostring(), colorfmt='rgba')

        # colorbar text generation
        self.cbtxt.text = '\n' + '\n'.join(('%.3e' % val for val in
                                            reversed(SCALAR_MAP.norm.inverse(np.linspace(0, 1, self.cbar_num)))))
        self.cbtxt.line_height = self.cbsize[1] / (self.cbar_num - 1) / (self.cbtxt.font_size + 3)
        self.cbtxt.center_y = 50 + self.cbsize[1] / 2 + self.cbtxt.font_size / 2

        # colorbar generation
        SCALAR_MAP.set_norm(colors.NoNorm())
        cb_raw = np.empty(self.cbsize[::-1])
        cb_raw[:] = np.expand_dims(np.linspace(0, 1, self.cbsize[1]), 1)
        cb_data = SCALAR_MAP.to_rgba(cb_raw) * 255
        self.cbtex.blit_buffer(cb_data.astype('uint8').tostring(), size=self.cbsize, colorfmt='rgba')

    def save_image(self):
        output_name = tkFileDialog.asksaveasfilename(title='Image Array Filename')
        if not output_name:
            return
        self.rend.save_irender(output_name, self.raw_data)

    def save_spectra(self):
        output_name = tkFileDialog.asksaveasfilename(title='Spectra Array Filename')
        if not output_name:
            return
        if self.raw_spectra is None:
            self.rend.distance_per_pixel = self.distance_per_pixel
            self.rend.stepsize = self.stepsize
            self.rend.y_pixel_offset = self.y_pixel_offset
            self.rend.x_pixel_offset = self.x_pixel_offset
            data, dfreqs, ny0, _ = self.get_il_render()
            self.raw_spectra = (noisify_spectra(data, self.noise_snr), dfreqs, ny0)
        self.rend.save_ilrender(output_name, self.raw_spectra)

    def save_range(self):
        self._saverangedialog.rend_choice = None
        self._saverangedialog.open()

    def _renderrangefromdialog(self, srd, choice):
        snap_bounds = sorted((int(srd.slider_snapmin.value), int(srd.slider_snapmax.value)))
        snap_skip = int(srd.slider_snapskip.value)
        snap_range = range(snap_bounds[0], snap_bounds[1], snap_skip)
        channellist = self.channellist
        channel_ids = [channellist.index(lib.text) for lib in srd.channelselect.adapter.selection]
        save_loc = srd.savefilename.text
        save_loct = Template(save_loc)
        if len(snap_range) > 1 and '${num}' not in save_loc or len(channel_ids) > 1 and '${chan}' not in save_loc:
            ed = ErrorDialog()
            ed.errortext = 'Missing "${num}" or "${chan}" in file descriptor'
            ed.open()
            return

        orig_mode, orig_snap, orig_channel = self.rendermode, self.snap, self.channel

        # if spectra is chosen, choose mode that caches spectra
        if choice == 'il':
            self.rendermode = Mode.doppler_shift

        for snap in snap_range:
            self.snap = snap
            for channel_id in channel_ids:
                self.channel = channel_id
                save_file = save_loct.substitute(num=str(snap), chan=channellist[channel_id])

                self.update(False)

                if choice == 'il':
                    self.rend.save_ilrender(save_file, self.raw_spectra)
                elif choice == 'i':
                    # process spectra into raw data if necessary
                    if self.rendermode == Mode.doppler_shift:
                        self.raw_data = self.spect_analyzer.quad_regc()
                        self.raw_data *= -CC / 1e3 / self.spect_analyzer.center_freq  # convert to km/s
                    elif self.rendermode == Mode.width:
                        self.raw_data = self.spect_analyzer.fwhm()
                        self.raw_data *= CC / 1e3 / self.spect_analyzer.center_freq  # convert to km/s
                    elif self.rendermode == Mode.asym:
                        self.raw_data = self.spect_analyzer.split_integral_vel(self.asym_sep, self.asym_width, 2)
                        self.raw_data = self.raw_data[..., 1] - self.raw_data[..., 0]
                    self.rend.save_irender(save_file, self.raw_data)

        srd.dismiss()
        self.mode, self.snap, self.channel = orig_mode, orig_snap, orig_channel
        self.raw_data = self.raw_spectra = None
        self.update()

    def get_i_render(self):
        return self.rend.i_render(self.channel, self.azimuth, -self.altitude,
                                  opacity=self.rend_opacity, verbose=False)

    def get_il_render(self):
        return self.rend.il_render(self.channel, self.azimuth, -self.altitude, nlamb=self.nlamb,
                                   opacity=self.rend_opacity, verbose=False)


class SaveRangeDialog(Popup):
    def __init__(self, render_widget, **kwargs):
        self.render_widget = render_widget
        self.channellist = render_widget.channellist
        self.snap_range = render_widget.rend.snap_range
        super(SaveRangeDialog, self).__init__(**kwargs)


class ErrorDialog(Popup):
    pass


class RenderApp(App):
    def __init__(self, rend):
        super(RenderApp, self).__init__(use_kivy_settings=False)
        self.rend = rend

    def build(self):
        game = RenderGUI(self.rend)
        game.update()
        return game


def show_renderer(rend):
    app = RenderApp(rend)
    app.run()
