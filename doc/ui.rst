User Interface
##############

The goal of the user interface (UI) is to visualise the output data and the fitted response model. In addition a
variety of options for the user to select and restrict the displayed range of data as well as a control
interface for the fit parameters should be provided. The UI consists of the three major sections:

* axis / fit options
* graph
* filter options

The UI provides four different plot types, namely 1D, 2D, 2D contour and 3D plots. The functionality of the
UI includes coloring the markers and fits, display of errors (1D & 2D), display the response model
with its uncertainty (variable σ-confidence) in 1D & 2D and limiting the range of the displayed input variables.
A more detailed description of the functionality can be found below.


User documentation
******************

Starting of the UI
==================

The UI is started via the terminal with the following command:

.. code-block:: python

    profit ui

Then the dash app will be started and can be viewed in the web-browser of your choice under
`http://127.0.0.1:8050/ <http://127.0.0.1:8050/>`_ which equals `localhost <http://localhost:8050/>`_ at port 8050.

General structure
=================

In order to visualise the data and provide a as straightforward as possible user experience in adjusting the options
a three section layout is used. The sections are listed below and shown in the figure.

* axis / fit options
* graph
* filter options

.. figure:: doc/pics/layout.png
  :width: 600

  Layout of the user interface with the three major sections: *axis/fit options*, *graph* and *filter options*.

axis/fit options
----------------
In this section of the layout the following three different types of options can be controlled:

* graph type
* options for the axis (including color and error)
* options for the fit based on the response model

.. figure:: doc/pics/axis.png

  Different types of options in the axis/fit options section of the UI.
graph type
^^^^^^^^^^

With the first radiobutton on the top, the type of graph can be selected. The following options are available:

* 1D (scatter + line)
* 2D (scatter + surface)
* 2D contour
* 3D (scatter + isosurface)

The four graph types are shown below with sample data and a sample response model:

.. figure:: doc/pics/ex_1D_fit.png
  :width: 600
  :align: center

  Example of the UI for a 1D graph.

.. figure:: doc/pics/ex_2D_fit.png
  :width: 600
  :align: center

  Example of the UI for a 2D graph.

.. figure:: doc/pics/ex_2Dc.png
  :width: 600
  :align: center

  Example of the UI for a 2D contour graph.

.. figure:: doc/pics/ex_3D_fit.png
  :width: 600
  :align: center

  Example of the UI for a 3D graph.

axis options
^^^^^^^^^^^^

The section **axis options** contains all control options concerning the selection and the display of the data.
This includes the following options:

.. confval:: input-variables

  :type: dropdown
  :options: all input-variables
  | number of rows depending on graphtype

.. confval:: output-variable

  :type: dropdown
  :options: all output-variables

.. confval:: log

  :type: checkbox
  :default: deactivated
  | activation of log-scale for each variable

.. confval:: marker color

  :type: dropdown & checkbox
  :options: input-variables | output-variables | *OUTPUT*
  :default: activated
  :available: 1D | 2D | 2D contour
  | Option *OUTPUT* is always syncronised with the :confval:output-variable.
  .. figure:: doc/pics/color_dd.png
    :width: 400
    :align: center

  Example for the possible variables for the marker color consisting of the *OUTPUT*-option and all in- and
  output-variables.

.. confval:: error

  :type: dropdown & checkbox
  :options: output-variables
  :default: deactivated
  :available: 1D | 2D
  | In order to be able to display the error, the error must be included in the output-file in a separate output-variable (column).
  .. figure:: doc/pics/error_1D.png
    :width: 600
    :align: center

    Example of errorbars for a 1D graphtype.

**Example:**

.. figure:: doc/pics/ex_axis_opt_2D.png
  :width: 400
  :align: center

  Example of the axis options for a 2D graphtype.

fit options
^^^^^^^^^^^

The section **fit options** contains all the control options concerning the activation/display and basic configuration
for the response model (fit). This includes depending on the graphtype (see sec. dynamic options) the following:

.. confval:: display fit

  :type: checkbox
  :default: deactivated

.. confval:: multi-fit

  :type: dropdown
  :options: input-variables
  :available: 1D | 2D
  Select dimension (variable) along which the number of fits specified in :confval:`#fits` will be constructed (only relevant if :confval:`#fits` > ``1``)

.. confval:: #fits

  :type: input
  :default: 1
  :available: 1D | 2D | 3D
  | Number of constructed fits along the dimension (variable) specified in the :confval:`multi-fit`.
  | **Caution**: In 3D the top and bottom isosurface is possibly only partly visible. Workaround increase :confval:`#fits` by 2.

.. confval:: σ-confidence

  :type: input
  :defualt: 2
  :available: 1D | 2D
  | Width of confidence interval. Types of display:
  | **1D**: area around the fit line
  | **2D**: two additional surfaces under and above the fit surface

.. confval:: add noise covariance

  :type: checkbox
  :default: deactivated
  :available: 1D | 2D
  | Takes uncertainty of underlying data into account for the response model.
  | Caution: Not supported for all surrogate models.

.. confval:: fit-color

  :type: radiobutton & checkbox
  :options: :confval:`output-variable` | :confval:`multi-fit` | :confval:`marker color`
  :default: output & activated
  :available: 2D

  | Controls dimension (variable) for the colorscale in 2D.
  | **1D**: same as :confval:`multi-fit`
  | **3D**: same as :confval:`output-variable`


.. confval:: fit-opacity

  :type: slider
  :range: [0%, 100%]
  :default: 50%
  :available: 1D | 2D | 3D
  | **1D**: opacity of area between upper and lower limit
  | **2D/3D**: opacity of all surfaces

.. confval:: #points

  :type: input
  :default: 50
  | Number of predictions for the fit out of the response model along the input axis.

Depending on the graphtype the fit will be a line (1D), a surface (2D) or an isosurface (3D).
The details how the parameters for the fits are selected can be found below in section *response model/fit*.


dynamic options
"""""""""""""""

When changing the graph type not only the graph changes but also all non relevant options disappear respectively
all relevant not visible options will be made visible. The behaviour is implemented for the following options:

* **axis options**:
    * input-variables (1D: x; 2D: x,y; 3D: x,y,z)
    * marker color (1D, 2D, 2D contour)
    * error (1D, 2D)
* **fit options**:
    * display fit (1D, 2D, 3D)
    * variable for multi-fit (1D, 2D)
    * #fits (1D, 2D, 3D)
    * σ-confidence (1D, 2D)
    * fit-color (2D)
    * fit-opacity (1D, 2D, 3D)

**Example:** number of input-variables

In *1D* at the **axis options** section only the input-variables for **x** will be shown because only one input-variable
is needed. If graph-type is switches to *2D* in addition a row for **y** will be visible because now two
input-variables are needed. The equivalent behaviour is implemented for the *3D* option.

.. figure:: doc/pics/graphtype1D.png
  :width: 400
  :align: center

  axis options: input-variable dropdowns (only x) for graphtype 1D

.. figure:: doc/pics/graphtype2D.png
  :width: 400
  :align: center

  axis options: input-variable dropdowns (x and y) for graphtype 2D


graph
-----
The section graph contains the actual graph. Since the graph is generated out of the plotly-library all the plotly
tools are available in the upper right corner. This tools include a png-download, zoom, pan, box and lasso select,
zoom in/out, autoscale, reset axis and various hover/selection tools.

There are different specific properties of the different graphtypes described below. In all graphtypes the axis
have the title according to the selected variable.

1D
^^
The 1D graph offers a range-slider beneath the plot. With the range-slider the displayed range of data can be defined
and moved along the axis. The alternative to the range-slider is to click&drag in the graph to select a certain
area, but with this method the viewed area can only be decreased.

2D/3D
^^^^^
In the 2D and 3D graph the graph can be rotated an tilted by click&drag. Unfortunately the camera positions resets as
soon as an option is changed.

2D contour
^^^^^^^^^^
In the 2D contour plot one fit is displayed. In addition all points in this area are also displayed. This can be quite
confusing because some points may have quite different values for the parameters (variables) not attached to the axis.
Therefore a narrowing of the range the non-axis parameters is recommended.

filter options
--------------
The third major section of the user interface are the filter options. The main function of the filter options is to
limit the range of the input-variables for the display in the plot and the determination of the parameters for the
prediction of the fit based on the response model.

The filter options are designed as a table. The controls for the entries of the table are located at the table
head and consist of the following:

* dropdown to select the input-variable to interact
* button "add filter" to add selected dropdown-option to table
* button "clear filter" to remove selected dropdown-option from table
* button "clear all" to remove all filters from table
* slider to select a scaling factor for the span of all filters
* button "scal filter span" to multiply the scaling factor of span onto the filter-span

If an variable is added to the filter table a new row appears in the table. The table consists of the following
columns:

* Parameter:
    The name of the variable (dimension).
* log:
    Checkbox to activate logscale for the whole row. All numeric values and the slider will be transformed to
    logscale. Default: deactivated
* Slider:
    A slider to limit the range.
* Range (min/max):
    Two input-fields for the upper and lower limit of the range.
* center/span:
    Two input-fields for the center and the span of the range.
* filter active:
    Checkbox to activate/deactivate the filter. Default: activated
* #digets:
    A input-field for the number of digits used for the calculation and display.
* reset:
    A button to reset the range to the default values (minimum to maximum).

Changes to the slider, the range-inputs or the center/span-inputs will automatically trigger a recalculation of the
other values. Also changes to the log-Checkbox or the #digets-input will be evaluated immediately.

In addition the center values determines the value of the parameter used for the prediction of the fit based on
the response model. If several fits along one dimension (variable) are predicted (**#fits** > 1), the minimum and
maximum of the range will be used for the limit of the linspace or logspace based on the log-Checkbox.
For details see section ().


Response model/fit
******************

For the prediction of the fit out of the response model the response model needs to be evaluated at different
places in the multidimensional parameter space. Therefore a multidimensional meshgrid is generated.
Along the dimensions of the plot (axis-variables) the meshgrid has the same length as the **#points**-input-field
specifies. The point are either linear-spaced or log-spaced based on the activation status of the **log**-checkbox
in the **axis options**-section beside the according dimension. The limits are based on the limit of the variables
respectively on the limits set in the filter-table.

In case of a single fit all non-axis parameters for the response model are constant. Where the center of the range
of this dimension is used. If the range is limited via the filer-table the fit adjusts accordingly.

In case of a multi-fit (**#fits** > 1) along a dimension (selected via **multi-fit**-dropdown) the minimum and
maximum of the range of this dimension will be used as limits for the generation of the vector. The number of points
is chosen according to the **#fits**-input-field. Restrictions of the limits via the filter-table will be taken into
account. Based on the activation-status of the **log**-checkbox in the filter-table a ``linspace`` or
``logspace``-vector is used.

For further details on the generation of the response model itself see the API documentation of the surrogate model.


Technical Background
********************

The User Interface (UI) is based on ``plotly/dash`` for Python (see `Homepage <https://dash.plotly.com/>`_).
Dash is a declarative and reactive web-based application. Dash is build on top
of the following components:

* Flask
* React.js
* Plotly.js

The entire UI is running on a ``Flask`` web server. Flask is a WSGI (Web Server Gateway
Interface) web app framework. When starting the ``Dash`` application a local webserver is
started via ``Flask``. It is possible to extend the application via Flask Plugins.

In ``Dash`` one is able to use the entire set of the ``plotly`` library. The frontend is
rendered via ``react.js`` (`react.js on github <https://github.com/facebook/react/>`_). ``react.js`` is a
declarative, component-based JavaScript library for building user interfaces developed an maintained by Facebook.

When working with ``Dash`` there are a lot of standard components available for the user via the
``dash_html_components`` library (see `dash_core_components on github <https://github.com/plotly/dash-core-components>`_) maintained by
the Dash core team. In addition it is possible to write your own component library via the standard open-source
React-to-Dash toolchain.

The second important library especially for structuring the UI is the ``dash_html_component`` library
(see `dash_html_component on github <https://github.com/plotly/dash-html-components>`_). It includes a set of HTML tags which are also
rendered via ``react.js``.

For customization it is possible to use ``CSS`` stylesheets for the entire interface as well as individual
``CSS``-styles for each element.

The graphs itself is based on the above mentioned ``plotly.js`` library
(see `github <https://github.com/plotly/plotly.js>`_). This graphic library maintained by Plotly.

