* Next
  - Added convenience centre of rotation methods to `AcquisitionGeometry` class.
    - `get_centre_of_rotation()` calculates the centre of rotation of the system
    - `set_centre_of_rotation()` sets the system centre of rotation with an offset and angle
    - `set_centre_of_rotation_by_slice()` sets the system centre of rotation with offsets from two slices

* 22.1.0
  - use assert_allclose in test_DataContainer
  - added multiple colormaps to show2D
  - Fix segfault in GradientOperator due to parameter overflows on windows systems
  - Fix angle display precision and matplotlib warning for sinograms with show2D

* 22.0.0
  - Strongly convex functionality in TotalVariation and FGP_TV Functions
  - Refactored KullbackLeibler function class. Fix bug on gradient method for SIRF objects
  - Numba added as a CIL requirement
  - Simplify initialisation of `CentreOfRotation.ImageSharpness` with new parameter `backend`
  - Added ISTA algorithm. Improve inheritance of proximal gradient algorithms
  - Updated interface to `plugins.tigre`/`plugins.astra` `FBP` and `ProjectionOperator` classes
  - Update NikonDataReader to parse and set up geometry with: `ObjectTilt` `CentreOfRotationTop` and `CentreOfRotationBottom`
  - Cleaned up unit test structure and output
  - Removal of deprecated code:
    - AcquisitionGeometry `__init__` no longer returns a configured geometry, use factory `create` methods instead
    - `subset` method removed, use `get_slice` or `reorder` methods
    - NikonDataReader `normalize` kwarg removed, use `normalise`
    - Algorithms initialisation `x_init` kwarg removed, use `initial`
    - Removed deprecated numpy calls
  - DataProcessors use weak-reference to input data
  - Merged CIL-ASTRA code in to CIL repository simplifying test, build and install procedures
    - Modules not moved should be considered deprecated
    - CIL remains licensed as APACHE-2.0
    - Minor bug fixes to the CPU 2D Parallel-beam FBP
  - Add ndim property for DataContainer class
  - Fixes show_geometry compatibility issue with matplotlib 3.5
  - Added ZEISSDataReader with cone/parallel beam, slicing, TXM Functionality
  - Raise exception if filename or data haven't been set in NexusDataWriter
  - Fixes error when update_objective_interval is set to 0 in an algorithm run
  - Deprecated:
    - TXRMDataReader is deprecated in favour of ZEISSDataReader 
  - GitHub Actions:
    - Update to version 0.1.1 of lauramurgatroyd/build-sphinx-action for building the documentation - ensures docs are always built from cil master
    
* 21.4.1
 - Removed prints from unittests and cleanup of unittest code.
 - CMake: 
   - install script re-allows selection of non default install directory ([#1246](https://github.com/TomographicImaging/CIL/issues/1246))
 - TIFF writer uses logging
 - Added unittests for TIFF functionality

* 21.4.0
  - PEP 440 compliant version
  - CMake fix due to use of pip install. 
  - Recon.FBP allows 'astra' backend 
  - Fixed PowerMethod for square/non-square, complex/float matrices with stopping criterion.
  - CofR image_sharpness improved for large datasets
  - Geometry alignmentment fix for 2D datasets
  - CGLS update for sapyb to enable complex data, bugfix in use of initial 
  - added sapyb and deprecated axpby. All algorithm updated to use sapyb.
  - Allow use of square brackets in file paths to TIFF and Nikon datasets

* 21.3.1
  - Added matplotlib version dependency to conda recipe
  - Fixed TIGRE wrappers for geometry with a virtual detector
  - Fixed TIGRE wrappers for cone-beam geometry with tilted rotation axis

* 21.3.0
  - Accelerated PDHG which handles strong convexity of functions
  - TotalVariation Function handles SIRF ImageData
  - Simulated datasets and volume added to DataExamples
  - TIGRE wrappers for parallel-beam geometry added
  - NEXUSWriter and NEXUSReader offer (8bit and 16bit) compression of data
  - show2D show_geom now return an object that can be saved with a `save` method
  - GradientOperator can be now used with SIRF DataContainers, both PET and MR
  - Add anisotropy in TotalVariation function
  - CCPi Regularisation plugin is refactored, only FGP_TV, FGP_dTV, TGV and TNV are exposed. Docstrings and functionality unit tests are added. Tests of the functions are meant to be in the CCPi-Regularisation toolkit itself.
  - Add dtype for ImageGeometry, AcquisitionGeometry, VectorGeometry, BlockGeometry
  - Fix GradientOperator to handle pseudo 2D CIL geometries
  - Created recon module with FBP and FDK using fast filtering library and TIGRE backprojectors
  - Added Intel IPP based library for filtering step of FBP
  - PDHG memory optimisation
  - ScaledFunction memory Optimisation
  - The github actions are merged into one action with multiple jobs
  - The conda build job uploads an artifact of the build tar.bz file which is later used by the documentation build job - which installs the package into a miniconda environment.
  - Documentation pages for recon, astra and cil-plugins are published.

* 21.2.0
  - add version string from git describe
  - add CCPi-Regularisation toolkit in unittests
  - show_geometry implemented to display AcquisitionGeometry objects, can be imported from utilities.display
  - CentreOfRotationCorrector.image_sharpness implemented which finds the rotation axis offset by maximising sharpness of a single slice reconstruction
  - Renamed CentreOfRotationCorrector.xcorr to CentreOfRotationCorrector.xcorrelation
  - Implemented Padder processor

* 21.1.0
  - Added TomoPhantom plugin to create 2D/3D + channel ImageData phantoms based on the TomoPhantom model library
  - Fixed bug in Zeiss reader geometry direction of rotation
  
* 21.0.0
  - Show2D now takes 4D datasets and slice infomation as input
  - TIGRE reconstruction package wrapped for cone-beam tomography
  - Datacontainers have get_slice method which returns a dataset with a single slice of the data
  - Datacontainers have reorder method which reorders the data in memory as requested, or for use with 'astra' or 'tigre'
  - Subset method has been deprecated
  - AcquisitionData and ImageData enforce requirement for a geometry on creation
  - New processors AbsorptionTransmissionConverter and TransmissionAbsorptionConverter to convert between Absorption and Transmission
  - Implemented Binner and Slicer processors
  - Implemented MaskGenerator and Masker processors

* 20.11.2
  - fixed windows build
  - NikonDataReader converts Nikon geometry to CIL geometry from xtekct file including detector and centre-or-rotation offsets
  - NexusdataReader supports files written with old versions of NexusDataWriter

* 20.11
  - python module renamed to cil
  - renamed Identity->IdentityOperator, Gradient->GradientOperator, SymmetrisedGradient->SymmetrisedGradientOperator

* 20.09.1
  - FiniteDifferenceOperator takes into consideration voxel size
  - Added CentreOfRotationCorrector
  - Removed CenterOfRotationFinder
  - moved TestData to utilities and renamed as dataexample
  - verbosity of Algorithms is independent of the update_objective_interval
  - added unittests
  - renamed
    - GradientDescent to GD
    - SparseFiniteDiff to SparseFiniteDifferenceOperator
    - LinearOperatorMatrix to MatrixOperator
  - bugfix update_objective of SPDHG 

* 20.09
  - added SPDHG algorithm
  - added TotalVariation function
  - Redesign of the AcquisitionGeometry class allowing more general acquisition trajectories than currently possible.
  - Added ZEISS reader

* 20.04
  - Significant upgrades to the operator and function classes to allow more flexible definition of optimisation problems
  - Added multithreaded C library for calculation of finite difference and some data processing
  - Added Gradient operator using C library and numpy backends

* 19.10
  - Improved usability with reader/writers and plotting utilities
  - Substantially improved test coverage

* 19.07
  - Introduction of BlockFramework
  - major revision and restructuring of the whole code
  - rewritten io package

* 19.02
  - introduction of Algorithm class
  - unit test expanded and moved to test directory
  - unified build system on Jenkins based on CCPi-VirtualMachine repo
 - switched to calendar versioning YY.0M.

* 0.10.0

* 0.9.4
  - Initial release
