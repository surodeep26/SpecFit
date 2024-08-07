import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from specutils.manipulation import gaussian_smooth
from specutils import Spectrum1D
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning

plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 14

warnings.filterwarnings('ignore', category=FITSFixedWarning)

def plot_spectrum_with_sliders(star_spectrum_file):
    '''    # Example usage:
        plot_spectrum_with_sliders('./observed_stars/vHD14143i.fits')'''


    # Directory containing the spectra files
    spectra_directory = './norm_tlusty'

    # Read all spectra files in the directory
    spectra_files = [file for file in os.listdir(spectra_directory) if file.endswith('.fits')]
    spectra_files.sort()  # Sort the files for consistent ordering

    # Extract temperature and gravity values from filenames
    temperatures = sorted(list(set([int(file.split('g')[0]) for file in spectra_files])))
    gravities = sorted(list(set([int(file.split('g')[1].split('v2')[0]) for file in spectra_files])))

    # Load spectra only for existing combinations of temperature and gravity
    spectra = {}
    for temperature in temperatures:
        spectra[temperature] = {}
        for gravity in gravities:
            filename = f"{temperature}g{gravity}v2.fits"
            if filename in spectra_files:
                spectra[temperature][gravity] = Spectrum1D.read(os.path.join(spectra_directory, filename))

    # Initial values for temperature, gravity, and smoothing
    initial_temperature = temperatures[0]
    initial_gravity = gravities[0]
    initial_smoothing = 1
    current_spectrum = spectra[initial_temperature][initial_gravity]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(
        top=0.965,
        bottom=0.23,
        left=0.03,
        right=0.99,
        hspace=0.2,
        wspace=0.2
    )

    line, = ax.plot(current_spectrum.spectral_axis, current_spectrum.flux, lw=2, color='red', alpha=0.5, label='Template')
    # Plot some general lines
    Hlines = {r'$\mathrm{H}_{\alpha}$': 6562.79, r'$\mathrm{H}_{\beta}$': 4861.35, r'$\mathrm{H}_{\gamma}$': 4340.47, r'$\mathrm{H}_{\delta}$': 4101.73, r'$\mathrm{H}_{\epsilon}$': 3970.07}
    color_map = plt.get_cmap('Spectral_r')
    min_wavelength = 4000
    max_wavelength = 7000
    for name, Hline in Hlines.items():
        normalized_wavelength = (Hline - min_wavelength) / (max_wavelength - min_wavelength)  # Adjust min and max accordingly
        ax.vlines(Hline, ymin=-.5, ymax=0.4, color=color_map(normalized_wavelength), alpha=0.7, linestyles='dashdot', lw=2)
        ax.text(Hline, 0, name, rotation=0, va='bottom', ha='center', fontsize=20)
 
    # Load the star spectrum
    star_spectrum = Spectrum1D.read(star_spectrum_file)
    star_line, = ax.plot(star_spectrum.spectral_axis, star_spectrum.flux, color='blue', alpha=0.7, label=f"{star_spectrum_file.split('/')[-1][1:-6]}")

    plt.title(star_spectrum_file.split('/')[-1][1:-6])
    plt.xlabel(r'Wavelength $\AA$')
    plt.ylabel(r'Flux')
    plt.xlim(4000, 4500)
    plt.ylim(0, 1.2)

    axcolor = 'red'

    # Define axes positions

    ax_smoothing_star = plt.axes([0.2, 0.10, 0.5, 0.03], facecolor=axcolor)
    ax_smoothing_spec = plt.axes([0.2, 0.07, 0.5, 0.03], facecolor=axcolor)
    ax_temp = plt.axes([0.2, 0.04, 0.5, 0.03], facecolor=axcolor)
    ax_gravity = plt.axes([0.2, 0.01, 0.5, 0.03], facecolor=axcolor)
    ax_radial_velocity = plt.axes([0.2, 0.14, 0.5, 0.03], facecolor=axcolor)

    slider_temp = Slider(ax_temp, 'Temp. ($K$)', temperatures[0], temperatures[-1], valinit=initial_temperature,
                         valstep=1000)
    slider_gravity = Slider(ax_gravity, r'$\log{g}$', gravities[0], gravities[-1], valinit=initial_gravity, valstep=25)
    slider_smoothing_spec = Slider(ax_smoothing_spec, 'smooth (template)', 1, 200, valinit=initial_smoothing, valstep=5)
    slider_smoothing_star = Slider(ax_smoothing_star, 'smooth (star)', 1, 10, valinit=initial_smoothing, valstep=0.5)
    slider_radial_velocity = Slider(ax_radial_velocity, 'Radial Velocity (km/s)', -50, 50, valinit=0, valstep=2)

    def update(val):
        temperature = int(slider_temp.val)
        gravity = int(slider_gravity.val)
        smoothing_spec = (slider_smoothing_spec.val)
        smoothing_star = (slider_smoothing_star.val)
        radial_velocity = slider_radial_velocity.val

        try:
            current_spectrum = spectra[temperature][gravity]
            # Apply smoothing to the spectrum
            smoothed_spectrum = gaussian_smooth(current_spectrum, smoothing_spec)
            line.set_ydata(smoothed_spectrum.flux)
            line.set_xdata(smoothed_spectrum.spectral_axis)

            # Apply radial velocity shift to the star spectrum
            shifted_star_spectrum = star_spectrum.shift_spectrum_to(radial_velocity=-radial_velocity  * u.Unit("km/s"))
            star_line.set_ydata(star_spectrum.flux)
            star_line.set_xdata(star_spectrum.spectral_axis)

            # Apply smoothing to the star spectrum
            smoothed_star_spectrum = gaussian_smooth(star_spectrum, smoothing_star)
            star_line.set_ydata(smoothed_star_spectrum.flux)
            star_line.set_xdata(smoothed_star_spectrum.spectral_axis)



            # Remove previous annotation text
            for txt in ax.texts:
                txt.remove()

            # Update text annotation
            ax.text(0.95, 0.05, f'Temperature: {temperature} K\n $\log{{g}}$: {gravity/100} dex \n RV: {radial_velocity} km/s',
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5))
        except KeyError:
            pass  # Do nothing if the spectra for the given temperature and gravity don't exist
        plt.legend()
        plt.draw()

    slider_temp.on_changed(update)
    slider_gravity.on_changed(update)
    slider_smoothing_spec.on_changed(update)
    slider_smoothing_star.on_changed(update)
    slider_radial_velocity.on_changed(update)


    # Function to handle keyboard events
    def on_key(event):
        if event.key == 'left':
            slider_temp.set_val(max(slider_temp.val - slider_temp.valstep, slider_temp.valmin))
        elif event.key == 'right':
            slider_temp.set_val(min(slider_temp.val + slider_temp.valstep, slider_temp.valmax))
        elif event.key == 'down':
            slider_gravity.set_val(max(slider_gravity.val - slider_gravity.valstep, slider_gravity.valmin))
        elif event.key == 'up':
            slider_gravity.set_val(min(slider_gravity.val + slider_gravity.valstep, slider_gravity.valmax))

    # Connect the keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()



def plot_spectrum_with_sliders_k(star_spectrum_file):
    '''    # Example usage:
        plot_spectrum_with_sliders('./observed_stars/vHD14143i.fits')'''


    # Directory containing the spectra files
    import os
    spectra_directory = './Kuruczall'

    # Read all spectra files in the directory
    spectra_files = [file for file in os.listdir(spectra_directory) if file.endswith('.fits')]
    spectra_files.sort()  # Sort the files for consistent ordering

    # Extract temperature and gravity values from filenames
    temperatures = sorted(list(set([int(file.split("_")[0]) for file in spectra_files])))
    gravities = sorted(list(set([float(file.split('_')[1].split(".fits")[0]) for file in spectra_files])))
    gravities_ = [f"{value:.5f}" for value in gravities]
    # Load spectra only for existing combinations of temperature and gravity
    spectra = {}
    for temperature in temperatures:
        spectra[temperature] = {}
        for gravity in gravities:
            filename = f"{temperature}_{gravity:.5f}.fits"
            if filename in spectra_files:
                spectra[temperature][gravity] = Spectrum1D.read(os.path.join(spectra_directory, filename))

    # Initial values for temperature, gravity, and smoothing
    initial_temperature = temperatures[0]
    initial_gravity = gravities[0]
    initial_smoothing = 1
    current_spectrum = spectra[initial_temperature][initial_gravity]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(
        top=0.965,
        bottom=0.23,
        left=0.03,
        right=0.99,
        hspace=0.2,
        wspace=0.2
    )

    line, = ax.plot(current_spectrum.spectral_axis, current_spectrum.flux, lw=2, color='red', alpha=0.5, label='Template')
    # Plot some general lines
    Hlines = {r'$\mathrm{H}_{\alpha}$': 6562.79, r'$\mathrm{H}_{\beta}$': 4861.35, r'$\mathrm{H}_{\gamma}$': 4340.47, r'$\mathrm{H}_{\delta}$': 4101.73, r'$\mathrm{H}_{\epsilon}$': 3970.07}
    color_map = plt.get_cmap('Spectral_r')
    min_wavelength = 4000
    max_wavelength = 7000
    for name, Hline in Hlines.items():
        normalized_wavelength = (Hline - min_wavelength) / (max_wavelength - min_wavelength)  # Adjust min and max accordingly
        ax.vlines(Hline, ymin=-.5, ymax=0.4, color=color_map(normalized_wavelength), alpha=0.7, linestyles='dashdot', lw=2)
        ax.text(Hline, 0, name, rotation=0, va='bottom', ha='center', fontsize=20)
 
    # Load the star spectrum
    star_spectrum = Spectrum1D.read(star_spectrum_file)
    star_line, = ax.plot(star_spectrum.spectral_axis, star_spectrum.flux, color='blue', alpha=0.7, label=f'{star_spectrum_file.split("/")[-1][1:-6]}')

    plt.title(star_spectrum_file.split('/')[-1][1:-6])
    plt.xlabel(r'Wavelength $\AA$')
    plt.ylabel(r'Flux')
    plt.xlim(4000, 4500)
    plt.ylim(0, 1.2)

    axcolor = 'red'

    # Define axes positions

    ax_smoothing_star = plt.axes([0.2, 0.10, 0.5, 0.03], facecolor=axcolor)
    ax_smoothing_spec = plt.axes([0.2, 0.07, 0.5, 0.03], facecolor=axcolor)
    ax_temp = plt.axes([0.2, 0.04, 0.5, 0.03], facecolor=axcolor)
    ax_gravity = plt.axes([0.2, 0.01, 0.5, 0.03], facecolor=axcolor)
    ax_radial_velocity = plt.axes([0.2, 0.14, 0.5, 0.03], facecolor=axcolor)

    slider_temp = Slider(ax_temp, 'Temp. ($K$)', temperatures[0], temperatures[-1], valinit=initial_temperature,
                         valstep=250)
    slider_gravity = Slider(ax_gravity, r'$\log{g}$', gravities[0], gravities[-1], valinit=initial_gravity, valstep=0.5)
    slider_smoothing_spec = Slider(ax_smoothing_spec, 'smooth (template)', 1, 200, valinit=initial_smoothing, valstep=2)
    slider_smoothing_star = Slider(ax_smoothing_star, 'smooth (star)', 1, 10, valinit=initial_smoothing, valstep=0.5)
    slider_radial_velocity = Slider(ax_radial_velocity, 'Radial Velocity (km/s)', -50, 50, valinit=0, valstep=2)

    def update(val):
        temperature = int(slider_temp.val)
        gravity = int(slider_gravity.val)
        smoothing_spec = (slider_smoothing_spec.val)
        smoothing_star = (slider_smoothing_star.val)
        radial_velocity = slider_radial_velocity.val

        try:
            current_spectrum = spectra[temperature][gravity]
            # Apply smoothing to the spectrum
            smoothed_spectrum = gaussian_smooth(current_spectrum, smoothing_spec)
            line.set_ydata(smoothed_spectrum.flux)
            line.set_xdata(smoothed_spectrum.spectral_axis)

            # Apply radial velocity shift to the star spectrum
            shifted_star_spectrum = star_spectrum.shift_spectrum_to(radial_velocity=-radial_velocity  * u.Unit("km/s"))
            star_line.set_ydata(star_spectrum.flux)
            star_line.set_xdata(star_spectrum.spectral_axis)

            # Apply smoothing to the star spectrum
            smoothed_star_spectrum = gaussian_smooth(star_spectrum, smoothing_star)
            star_line.set_ydata(smoothed_star_spectrum.flux)
            star_line.set_xdata(smoothed_star_spectrum.spectral_axis)



            # Remove previous annotation text
            for txt in ax.texts:
                txt.remove()

            # Update text annotation
            ax.text(0.95, 0.05, f'Temperature: {temperature} K\n $\log{{g}}$: {gravity/100} dex \n RV: {radial_velocity} km/s',
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5))
        except KeyError:
            pass  # Do nothing if the spectra for the given temperature and gravity don't exist
        plt.draw()

    slider_temp.on_changed(update)
    slider_gravity.on_changed(update)
    slider_smoothing_spec.on_changed(update)
    slider_smoothing_star.on_changed(update)
    slider_radial_velocity.on_changed(update)


    # Function to handle keyboard events
    def on_key(event):
        if event.key == 'left':
            slider_temp.set_val(max(slider_temp.val - slider_temp.valstep, slider_temp.valmin))
        elif event.key == 'right':
            slider_temp.set_val(min(slider_temp.val + slider_temp.valstep, slider_temp.valmax))
        elif event.key == 'down':
            slider_gravity.set_val(max(slider_gravity.val - slider_gravity.valstep, slider_gravity.valmin))
        elif event.key == 'up':
            slider_gravity.set_val(min(slider_gravity.val + slider_gravity.valstep, slider_gravity.valmax))

    # Connect the keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
    plt.legend()




def plot_spectrum(spec, smooth=None, labels=None, xlims=(4000,7000), save=None):
    if smooth is None:
        smooth = [1]
        smooth = np.ones(len(spec))

    spec_s = []
    for _spec, _smooth in zip(spec, smooth):
        spec_s.append(gaussian_smooth(_spec,_smooth))
    
    plt.figure(figsize=(20,4))
    plt.title(f'{labels[0]}')
    plt.xlabel(r'Wavelength $\AA$')
    plt.xlim(xlims[0],xlims[1])
    plt.ylim(0.25,1.5)
    plt.ylabel(r'Normalized Flux')
    plt.tight_layout()
    plt.grid(color='lightgrey')
    colors = ['black', 'red']
    als = [1,0.6]
    if labels is not None:
        for spectra,labl,clr,al in zip(spec_s, labels, colors,als):
            plt.plot(spectra.spectral_axis, spectra.flux, alpha=al, label=labl, c=clr)
            plt.legend(loc='upper right')
    if labels is None:
        for spectra in spec_s:
            plt.plot(spectra.spectral_axis, spectra.flux, alpha=0.8)

    if save is not None:
        # Split the save variable to get the first part of the filename
        save_prefix = save.split('_')[0]
        for file in os.listdir('./observed_stars/'):
            # Check if the file matches the prefix and has a .pdf extension
            if file.startswith(save_prefix) and file.endswith('.pdf'):
                print('A pdf for the star exists')
                # Construct the full file path
                file_path = os.path.join('./observed_stars',file)
                # Delete the existing file
                os.remove(file_path)
                print(f'Deleted existing file: {file_path}')
                break  # Stop checking once the file is found and deleted
        
        # Save the new file
        plt.savefig(f'./observed_stars/{save}.pdf')
        print(f'Saved new file: ./observed_stars/{save}.pdf')
    return plt