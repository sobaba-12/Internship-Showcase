filename = "assessment_data_py22mat.dat"
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import j1
from matplotlib import cm, ticker

def ProcessData(filename):
    """At least this function must exist in your code and it mist have this def line and return a dictionary.

    Your function can make use of other functions, which you define within your code file.

    Your code must not use 'input' as there is no user to input data when the code is tested and
    anything printed to the screen cannot be assessed.

    You should replace this docstring with an appropriate one for your code."""

    # Your ProcessData code should go here.
    metadata, data = ReadData(filename)
    checkData(metadata)
    sin_axis, intensities, vertical0, horizontal0, diagonal2, diagonal2 = GetSlices(metadata, data)
    PlotDiffraction(data[0,1:], intensities) # Plot diffraction grating pattern
    mega_results = CurveFitting(sin_axis, vertical0, horizontal0, diagonal2, diagonal2, 2 * np.pi / (metadata['Wavelength (nm)'] * 1e-9)) # Results from curve-fitting
    
    # Initialise results
    I0 = mega_results["I0"]
    error_I0 = mega_results["Error in I0"]
    dim = mega_results["Dimension"]
    error_Dim = mega_results["Error in Dimension"]
    chi = mega_results["Chi-Squared"]
    
    shape = GetShape(dim, chi)    # Get the shape for the aperture
    dim1, dim2, error_Dim1, error_Dim2 = GetDimensions(dim, error_Dim, shape) # Get the dimensions and their respective errors for 1 and 2
    watt_Area, error_Area = GetIntensities(shape,I0,error_I0,dim,error_Dim) # Get the watt/area and respective error
    
    # This is the data structure to return your results with -
    # replace the None values with your answers. Do not
    # rename any of the keys, do not delete any of the keys
    # - if your code doesn't find a value, leave it as None here.
    results = {
        "shape": shape,
        # one of "square", "rectangle", "diamond", "circle" - must always be present.
        "dim_1": dim1,
        # a floating point number of the first dimension expressed in microns
        "dim_1_err": error_Dim1,
        # The uncertainty in the above, also expressed in microns
        "dim_2": dim2,
        # For a rectangle, the second dimension, for other shapes, the same as dim_1
        "dim_2_err": error_Dim2,
        # The uncertainty in the above, also expressed in microns
        "I0/area": watt_Area,
        # The fitted overall intensity value/area of the aperture.
        "I0/area_err": error_Area,  # The uncertainty in the above.
    }
    return results
def ReadData(filename):
    """
    Reads the data file and returns the metadata and actual data
    :param filename: [str] the filename used
    :return: metadata and data
    """
    metadata = dict()

    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()

                if line == "&END":
                    data = np.loadtxt(file,
                                      dtype=np.float64)  # skips the last "&END"
                    if data.ndim != 2:
                        raise RuntimeError(
                            "Data did not have rows and columns")
                    if data.shape[0] == 0:
                        raise RuntimeError("No rows of data found")
                    if data.shape[1] == 0:
                        raise RuntimeError("No columns of data found")
                    if data.shape[1] != data.shape[0]:
                        raise RuntimeError(
                            "The number of columns and rows do not match")
                    break

                if "=" in line:
                    parts = line.split('=')
                    if len(parts) != 2:
                        raise RuntimeError(
                            f"Unable to parse {line} as name=value")
                    try:
                        metadata[parts[0]] = float(parts[1])
                    except ValueError:
                        metadata[parts[0]] = parts[1]
    except IOError as err:
        raise IOError(
            f"Operating system error reading the file: original message was {str(err)}")
    except RuntimeError as err:
        raise IOError(
            f"Something wrong with the file format: message was {str(err)}")
    except ValueError as err:
        raise IOError(
            f"An error occurred reading the data with loadtxt: {str(err)}")

    return metadata, data

def checkData(metadata):
    """
    Does a simple check error to make sure that there are no zero or negative values
    :param metadata: [dict] the metadata read from the data
    :return: raises an error
    """
    dist = metadata["Distance to Screen (m)"]
    # Check for non-negative values
    if np.any(dist) < 0:
        raise ValueError("The distance to the screen cannot have a negative value")
    # Check for non-zero values to avoid division by 0
    if np.any(dist) == 0:
        raise ZeroDivisionError("The distance to the screen cannot be 0")

def GetSlices(metadata, data):
    """
    The different noisy slices that will be fitted to the various functions; since
    noisy data is being used the slices sin_theta/phi = 0 are made arbitrarily by
    choosing the column closest to 0
    :param metadata: [dict] the metadata of the read file
    :param data: [array] the data of the read file
    :return: sin_axis, intensities, vertical0, horizontal0, diagonal1, diagonal2
    """

    # Getting the right dimensions for our data
    axis = data[0, 1:]
    sin_axis = axis * 1e-3 / (np.sqrt(
        (axis * 1e-3) ** 2 + metadata['Distance to Screen (m)'] ** 2))

    # The various slices that will be fitted to functions
    intensities = data[1:, 1:]
    vertical0 = intensities[0:, 125]
    horizontal0 = intensities[125, 0:]
    diagonal1 = np.diag(intensities)
    diagonal2 = np.diag(np.fliplr(intensities))[::-1]
    return sin_axis, intensities, vertical0, horizontal0, diagonal1, diagonal2

def ChiSquared(o_i, f_i):
    """
    Goodness of fit test (lower is better)
    :param o_i: [1D array] measured data points
    :param f_i: [1D array] expected data points
    :return: chi-value
    """
    if not isinstance(o_i, (np.ndarray, float)):
        raise TypeError("The measured data points should be an array or a float")
    if not isinstance(f_i, (np.ndarray, float)):
        raise TypeError("The measured data points should be an array or a float")

    squ_sum = 0
    for i in range(0, 250):
        squ_sum += (f_i[i] - o_i[i]) ** 2
    chi_squ = squ_sum / (len(o_i) + 1)
    return chi_squ

def PlotDiffraction(axis, intensities):
    """
    Plots the diffraction pattern from the data
    :param axis: [1D array] the horizontal and vertical axes of the data
    :param intensities: [array] the intensities of the read data from the file
    :return: the figure object
    """
    # Error catching
    if not isinstance(axis, np.ndarray):
        raise TypeError("Axis should be a numpy array")
    if not isinstance(intensities, np.ndarray):
        raise TypeError("The intensities should be a numpy array")

    # Diffraction Pattern
    fig1, ax1 = plt.subplots()
    Axis1, Axis2 = np.meshgrid(axis, axis)
    ax1.set_title(
        "Diffraction pattern for an unknown diffraction grating py22mat")
    ax1.set_xlabel("mm")
    ax1.set_ylabel("mm")
    x = np.floor(np.log10(intensities.min()))
    cs = ax1.contourf(Axis1, Axis2, intensities, cmap=cm.plasma,
                     locator=ticker.LogLocator())
    cbar = fig1.colorbar(cs)
    cbar.ax.set_ylabel("Intensity")

def CurveFitting(sin_axis, vertical0, horizontal0, diagonal1, diagonal2, wave_number):
    """
    Initially plots the diffraction pattern generated by the data
    Then fits various functions to the strips
    :param sin_axis: [1D array] the sin axes of the data
    :param vertical0: [1D array] the middle vertical strip of data
    :param horizontal0: [1D array] the middle horizontal strip of data
    :param diagonal1: [1D array] one of the diagonals of the data
    :param diagonal2: [1D array] the other diagonal of the data
    :param wave_number: [float] the wave number calaculated from the metadata of the data
    :return: a dictionary with the values for I0, Dimensions 1 & 2 with the respective ChiSquared value
    """

    # Error catching
    if not isinstance(sin_axis, np.ndarray):
        raise TypeError("The sin axes of the data should be a numpy array")
    if not isinstance(vertical0, np.ndarray):
        raise TypeError("The middle vertical strip should be a numpy array")
    if not isinstance(horizontal0, np.ndarray):
        raise TypeError("The middle horizontal strip should be a numpy array")
    if not isinstance(diagonal1, np.ndarray):
        raise TypeError("The diagonal strip should be a numpy array")
    if not isinstance(diagonal2, np.ndarray):
        raise TypeError("The diaonal strip should be a numpy array")
    if not isinstance(wave_number, (float,int)):
        raise TypeError("The wave number should either an integer or a float")

    # Fitting functions (unnormalised sinc function)
    def I_strip(sin_angle, I0, W):
        """
        The intensity function for the horizontal/vertical slice
        :param sin_angle: angle of point [independent function] (float)
        :param I0: initial intensity (float)
        :param W: width/diameter/Dimension 1 []unsure about units but should be m (float)
        :return: The intensity
        """
        return I0 * np.sinc((wave_number * (W / 2) * sin_angle)) ** 2

    def I_diag(sin_angle, I0, W):
        """
        The intensity function for diamond shape
        :param sin_angle: diagonal slice [independent function] (float)
        :param I0: initial intensity (float)
        :param W: length of one side of diagonal (float)
        :return: The intensity
        """
        return I0 * np.sinc(
            wave_number * W * sin_angle / (2 * np.sqrt(2))) ** 4

    def I_circle(sin_angle, I0, D):
        """
        The intensity function for a circular aperture
        :param sin_angle: diagonal slice [independent function] (float)
        :param I0: initial intensity (float)
        :param D: diameter (float)
        :return: The intensity
        """
        return I0 * ((2 * j1(np.pi * wave_number * D * sin_angle)) / (
                np.pi * wave_number * D * sin_angle)) ** 2

    # Curve Fitting with data
    x = np.linspace(-.05, .05, 1000)
    f = np.linspace(np.min(sin_axis), np.max(sin_axis), 250)
    I0 = []
    error_I0 = []
    dim = []
    error_dim = []
    chi = []

    # Vert -> I_strip [0]
    popt1, pcov1 = curve_fit(I_strip, sin_axis, vertical0,
                             p0=[np.max(vertical0), 2e-5])
    perr1 = np.sqrt(np.diag(pcov1))
    I0.append(popt1[0])
    error_I0.append(perr1[0])
    dim.append(popt1[1])
    error_dim.append(perr1[1])
    chi.append(ChiSquared(vertical0, I_strip(f, *popt1)))

    # Horiz -> I_strip [1]
    popt2, pcov2 = curve_fit(I_strip, sin_axis, horizontal0,
                             p0=[np.max(horizontal0), 2e-5])
    perr2 = np.sqrt(np.diag(pcov2))
    I0.append(popt2[0])
    error_I0.append(perr2[0])
    dim.append(popt2[1])
    error_dim.append(perr2[1])
    chi.append(ChiSquared(horizontal0, I_strip(f, *popt2)))

    # Vert -> I_diag [2]
    popt3, pcov3 = curve_fit(I_diag, sin_axis, vertical0,
                             p0=[np.max(vertical0), 2e-5])
    perr3 = np.sqrt(np.diag(pcov3))
    I0.append(popt3[0])
    error_I0.append(perr3[0])
    dim.append(popt3[1])
    error_dim.append(perr3[1])
    chi.append(ChiSquared(vertical0, I_diag(f, *popt3)))

    # Horiz -> I_diag [3]
    popt4, pcov4 = curve_fit(I_diag, sin_axis, horizontal0,
                             p0=[np.max(horizontal0), 2e-5])
    perr4 = np.sqrt(np.diag(pcov4))
    I0.append(popt4[0])
    error_I0.append(perr4[0])
    dim.append(popt4[1])
    error_dim.append(perr4[1])
    chi.append(ChiSquared(horizontal0, I_diag(f, *popt4)))

    # Diag1 -> I_strip [4]
    popt5, pcov5 = curve_fit(I_strip, sin_axis, diagonal1,
                             p0=[np.max(diagonal1), 2e-5])
    perr5 = np.sqrt(np.diag(pcov5))
    I0.append(popt5[0])
    error_I0.append(perr5[0])
    dim.append(popt5[1])
    error_dim.append(perr5[1])
    chi.append(ChiSquared(diagonal1, I_strip(f, *popt5)))

    # Diag2 -> I_strip [5]
    popt6, pcov6 = curve_fit(I_strip, sin_axis, diagonal2,
                             p0=[np.max(diagonal2), 2e-5])
    perr6 = np.sqrt(np.diag(pcov6))
    I0.append(popt6[0])
    error_I0.append(perr6[0])
    dim.append(popt6[1])
    error_dim.append(perr6[1])
    chi.append(ChiSquared(diagonal2, I_strip(f, *popt6)))

    # Horz -> I_circle [6]
    popt7, pcov7 = curve_fit(I_circle, sin_axis, horizontal0,
                             p0=[np.max(horizontal0), 2e-5])
    perr7 = np.sqrt(np.diag(pcov7))
    I0.append((popt7[0]))
    error_I0.append((perr7[0]))
    dim.append(popt7[1])
    error_dim.append((perr7[1]))
    chi.append(ChiSquared(horizontal0, I_circle(f, *popt7)))

    # Vert -> I_circle [7]
    popt8, pcov8 = curve_fit(I_circle, sin_axis, vertical0,
                             p0=[np.max(vertical0), 2e-5])
    perr8 = np.sqrt(np.diag(pcov8))
    I0.append((popt8[0]))
    error_I0.append((perr8[0]))
    dim.append(popt8[1])
    error_dim.append((perr8[1]))
    chi.append(ChiSquared(vertical0, I_circle(f, *popt8)))

    # Plot for vertical strip:
    fig2, ax2 = plt.subplots()
    ax2.plot(sin_axis,vertical0, '.', label="data")
    ax2.plot(f, I_strip(f, *popt1), label="square/rectangle*")
    ax2.plot(f, I_diag(f, *popt3), label="diamond")
    ax2.plot(f, I_circle(f, *popt8), label="circle")
    ax2.legend()
    ax2.set_title("Slice through $\phi = 0$")
    ax2.set_xlabel("sin(angle)")
    ax2.set_ylabel("Intensities")
    ax2.annotate('$I_0 = 8.82 \pm 0.03 nm$\n$H = 21.69 \pm 0.08\mu m$', xy=(0.2, 0.7),
                 xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=12)
    plt.show()
    # Plot for horizontal strip:
    fig3, ax3 = plt.subplots()
    ax3.plot(sin_axis, horizontal0, '.', label="data")
    ax3.plot(f, I_strip(f, *popt2), label="square/rectangle*")
    ax3.plot(f, I_diag(f, *popt4), label="diamond")
    ax3.plot(f, I_circle(f, *popt7), label="circle")
    ax3.legend()
    ax3.set_title(r"Slice through $\theta = 0$")
    ax3.set_xlabel("sin(angle)")
    ax3.set_ylabel("Intensities")
    ax3.annotate('$I_0 = 8.83 \pm 0.03 nm$\n$W = 17.62 \pm 0.06\mu m$',
                 xy=(0.2, 0.7),
                 xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=12)
    plt.show()
    # Final results
    mega_results = {
        "I0": I0,
        "Error in I0": error_I0,
        "Dimension": dim,
        "Error in Dimension": error_dim,
        "Chi-Squared": chi
    }
    return mega_results

def GetShape(dim, chi):
    """
    Outputs the likely shape of the data given
    :param dim: [list] list of all the respective dimensions
    :param chi: [list] list of all the respective chi values
    :return: The shape of the final result
    """
    # Error catching
    if not isinstance(dim, (list,float)):
        raise TypeError("The dimensions should be a float or a list of floats")
    if not isinstance(chi, (list,float)):
        raise TypeError("The chi-squared values should be a float or a list of floats")

    # Initialise
    shape = ""

    dia_dim1, dia_dim2 = np.sqrt((dim[4] ** 2) / 2), np.sqrt((dim[5] ** 2) / 2)
    # Average values of

    # Check square/rectangle apertures
    if np.min(chi) in chi[0:2]:
        if abs(dim[1] - dim[0]) <= 1e-6:
            shape = "square"
        else:
            shape = "rectangle"
    else:
        if abs(dim[2] - dim[3]) and abs(dim[2] - dia_dim1) and abs(dim[3] - dia_dim2) <= 1e-6:
            shape = "diamond"
        else:
            shape = "circle"
    return shape

def errorAdd(*errors):
    """
    Calculates the absolute uncertainty when adding/subtracting values
    :param errors: [float] the absolute uncertainties
    :return: final error
    """
    # Error-catching
    """if not isinstance(errors, (float,int)):
        raise TypeError("Errors should be a number")"""

    square_error = 0
    for num in errors:
        square_error += num ** 2
    return np.sqrt(square_error)

def errorMul(dimensions, *errors):
    """
    Calculates the absolute uncertainty when multiplying/division values
    :param dimensions: [list] the associated intensities with the errors
    :param errors: [float] the absolute uncertainties
    :return: final error
    """
    # Error catching
    """if not isinstance(dimensions, (list, float)):
        raise TypeError("The dimensions should be a float or a list of floats")
    if not isinstance(errors, float):
        raise TypeError("Errors should be floats")"""

    square_error = 0
    i = 0
    for num in errors:
        square_error += (num / dimensions[i]) ** 2
        i += 1
    return np.sqrt(square_error)

def GetDimensions(dim, error_Dim, shape):
    """
    Returns the dimensions with their associated errors
    :param dim: [list] the respective calculated dimensions
    :param error_Dim: [list] the respective calculated errors in the dimensions
    :param shape: [str] the determined shape of the data
    :return: dim1, dim2, error_Dim1, error_Dim2
    """
    # Error catching
    if not isinstance(dim, (float,list)):
        raise TypeError("Dimensions should be a list of floats")
    if not isinstance(error_Dim, (float,list)):
        raise TypeError("The errors in the dimensions should be a list of floats")
    if not isinstance(shape, str):
        raise TypeError("The shape of the aperture should be a string")

    if shape in ["square", "rectangle"]:
        return dim[0], dim[1], error_Dim[0], error_Dim[1]
    elif shape in ["diamond"]:
        return dim[2], dim[3], error_Dim[2], error_Dim[3]
    else:
        D = np.average(dim[6:8])
        error_D = errorAdd(error_Dim[6], error_Dim[7]) / 2
        return D, D, error_D, error_D

def GetIntensities(shape, I0, error_I0, dim, error_dim):
    """
    Calculates the maximum intensity and the intensity per unit area
    :param shape: [str] the given shape of the data
    :param I0: [list] the respective calculated values of the intensity
    :param error_I0: [list] the respective calculated values of the errors in the intensity
    :param dim: [list] the respective calculated values of the dimensions
    :param error_dim: [list] the respective calculated values of the errors in the dimension
    :return: intensity per unit area, error in the intensity per unit area
    """
    # Error catching
    if not isinstance(I0, (list, float)):
        raise TypeError("The fitted intensities should be a list of floats")
    if not isinstance(error_I0, (list, float)):
        raise TypeError("The error in the fitted intensities should be a list of floats")


    # Get results
    if shape in ["square", "rectangle"]:
        max_I = np.average(I0[0:2])
        area = dim[0] * dim[1]
        error_I = errorAdd(error_I0[0], error_I0[1]) / 2
        error_Area = errorMul(dim[0:2], dim[0], dim[1])
    elif shape in ["diamond"]:
        max_I = np.average(I0[2:4])
        area = dim[2] * dim[3]
        error_I = errorAdd(error_I0[2], error_I0[3]) / 2
        error_Area = errorMul(dim[2:4], dim[2], dim[3])
    else:
        max_I = np.average(I0[6:8])
        R = np.average(dim[6:8]) / 2
        area = np.pi * (R) ** 2
        error_I = errorAdd(error_I0[6], error_I0[7]) / 2
        error_R = errorAdd(error_dim[6],error_dim[7]) / 4
        error_RSquared = errorMul([R, R], error_R, error_R)
        error_Area = error_RSquared * np.pi

    watt_Area = max_I / area
    return watt_Area, error_Area

if __name__=="__main__":
     # Put your test code in side this if statement to stop
     #it being run when you import your code
     filename = "assessment_data_py22mat.dat"
     test_results=ProcessData(filename)
     print(test_results)
     # check that values are within the correct range

# Issues with circle no noise data and random data ie. when it is a circle it sometimes thinks it's a diamond
# Just take the L with perfection on finding the right answer for random data 1 and circle no noise -> use them as example
