User Interface
##############

Idea
****

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
a three section layout is used. As shown in the figure below, the upper left column is used for the *axis/fit options*.
In the upper right column the *graph* itself is located. On the bottom of these two sections the *filter options*
section is placed. The functionality of each section will be described in the according sections below.

.. figure:: doc/pics/layout.png
  :width: 600

  Layout of the user interface with the three major sections.

axis/fit options
----------------
In this section of the layout three different types of options can be controlled, namely the graph type itself,
the options for the axis including the color and the error. In addition the different options for the fit based on
the response model.

graph type
^^^^^^^^^^

With the first radiobutton on the top, the type of graph can be selected. The following options are available:

* 1D
* 2D
* 2D contour
* 3D

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


axis options
^^^^^^^^^^^^

The section axis options contains all control options concerning the selection and the display of the data. This
includes the different in- and output-variables (number depending on graphtype), the marker color and the errorbars.
In the figure below the axis options section for the case of the 2D graphtype is shown.

.. figure:: doc/pics/ex_axis_opt_2D.png
  :width: 400
  :align: center

  Example of the axis options for a 2D graphtype.

in-/output-variables
""""""""""""""""""""

The up to three rows for the input-variables provided all inputs from the input-file as a dropdown option. The
output-variable dropdown provides the according option form the output-file. All rows provide a
**log**-checkbox which scale the according axis in the plot in logscale.

color
"""""

With the **color** option the marker-color can be controlled. The dropdown options include all in- and output-variables
and in addition the option *OUTPUT* in CAPS at the first position. This options is a reference to the selected
output-variable in the **output**-dropdown. Therefore the coloraxis will always be in sync with the output.

Furthermore the color can be activated/deactivated via the checkbox. The color is activated by default after loading
the UI.

.. figure:: doc/pics/color_dd.png
  :width: 400
  :align: center

  Example for the possible variables for the marker color consisting of the *OUTPUT*-option and all in- and
  output-variables.

error
"""""

The 1D and 2D plot support errorbars. In order to be able to display the error, the error must be included in the
output-file in a separate output-variable (column). Technically it the **error**-dropdown provides all output-variables
as dropdown options. With the checkbox the errorbars can be activated/deactivated and are deactivared by default.

.. figure:: doc/pics/error_1D.png
  :width: 600
  :align: center

  Example of errorbars for a 1D graphtype.

fit options
^^^^^^^^^^^

The section **fit options** contains all the control options concerning the activation/display and basic configuration
for the response model (fit). This includes depending on the graphtype (see sec. dynamic options) the following:

* checkbox for activation/display
* dropdown for the variable of the multi-fit
* input-field for the number of fits
* input-field for the σ-confidence
* checkbox to add the noise covariance
* radiobuttons for the fit-color
* slider for the fit-opacity
* input-field for the number of points

Depending on the graphtype the fit will be a line (1D), a surface (2D) or an isosurface (3D).
The details how the parameters for the fits are selected can be found below in section *Response model/fit*).

multi-fit
"""""""""
With the **multi-fit**-dropdown the user can select along which dimension (variable) the number of fits specified in
the **#fits**-input-field will be constructed (only relevant if **#fits** > 1). All input-variables are possible
dropdown-options.

#fits
"""""
With the **#fits**-input-field the number of constructed fits along the dimension (variable) specified in the
**multi-fit**-dropdown can be controlled. The default value is set to 1.

Caution: In 3D the top and bottom isosurface is possibly only partly visible, maybe **#fits** needs to be improved by 2
to actually see the desired number of fits.

σ-confidence
""""""""""""
With the **σ-confidence**-input-field the width of the confidence interval can be controlled. The confidence interval
is only available in 1D and 2D and the default value is set to 2. Depending on the graphtype the the confidence
interval is either displayed as area around the fit line (1D) or as two additional surfaces under an above the
fit surface (2D).

The checkbox *add noise covariance* takes the uncertainty of the underlying data for the response model into account.
This option is not available for every surrogate model, check Terminal for possible warning.

fit-color
"""""""""
With the **fit-color**-radiobutton the coloraxis of the fits can be controlled (dimension/variable for colorscale).
This options is only available in 2D. In 1D the fit-color is automatically defined as the **mulit-fit**-variable
and in 3D the fit-color is equal to the output-variable (in 3D already the color). For the 2D graphtype the
following options a provided:

* output
    sync with output-variable
* multi-fit
    sync with mutlti-fit-variable
* maker-color
    sync with color-variable

The output options is selected by default.

fit-opacity
"""""""""""
With the **fit-opacity**-slider the opacity of the surfaces respectively the confidence interval display can be
controlled. In 1D it controls the opacity of the area between the upper and lower limit, in 2D and 3D the opacity
of all surfaces. The default value is set to 50%.

#points
"""""""
With the **#points**-input-field the number of point for the prediction of the fit out of the response model can be
controlled. The default value is set to 50.


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



