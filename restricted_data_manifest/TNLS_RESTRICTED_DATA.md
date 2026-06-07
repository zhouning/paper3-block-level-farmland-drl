# Restricted Raw Data Manifest

The Paper3 experiments use parcel geometry and attributes from the Third
National Land Survey (TNLS) for Bishan District, China. These raw data are
governed by access restrictions and cannot be redistributed in this public
repository.

## Raw Data Not Included

- TNLS parcel geodatabase and derived parcel GeoPackage files.
- Township boundary shapefiles.
- DEM raster tiles and raster-derived NumPy arrays.
- Private Colab training packages that bundled restricted geospatial files.

## Expected Local Inputs for Authorized Users

Set these paths through environment variables:

- `PAPER3_DLTB_PATH`: parcel GeoPackage with TNLS attributes and `slope_mean`.
- `PAPER3_TNLS_GDB_PATH`: original or controlled-access TNLS geodatabase.
- `PAPER3_XIANGZHEN_PATH`: township boundary shapefile.
- `PAPER3_DEM_INTERMEDIATE_DIR`: DEM/slope intermediate directory.

## Public Substitute Artifacts

The repository includes derived block metrics, model outputs, training logs,
evaluation JSON files, and figures so that reported results can be audited
without redistributing raw parcel geometry.
