import os
import rasterio
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.vrt import WarpedVRT
from rasterio.mask import mask as rasterio_mask
from .utils import _check_rasterio_im_load, _check_crs, raster_get_projection_unit, split_geom, _check_gdf_load, get_projection_unit, split_multi_geometries, save_empty_geojson
import numpy as np
from shapely.geometry import box
from tqdm.auto import tqdm
from shapely.geometry import box, Polygon
import geopandas as gpd


class RasterTiler(object):
    """An object to tile geospatial image strips into smaller pieces.
    Arguments
    ---------
    dest_dir : str, optional
        Path to save output files to. If not specified here, this
        must be provided when ``Tiler.tile_generator()`` is called.
    src_tile_size : `tuple` of `int`s, optional
        The size of the input tiles in ``(y, x)`` coordinates. By default,
        this is in pixel units; this can be changed to metric units using the
        `use_src_metric_size` argument.
    use_src_metric_size : bool, optional
        Is `src_tile_size` in pixel units (default) or metric? To set to metric
        use ``use_src_metric_size=True``.
    dest_tile_size : `tuple` of `int`s, optional
        The size of the output tiles in ``(y, x)`` coordinates in pixel units.
    dest_crs : int, optional
        The EPSG code or rasterio.crs.CRS object for the CRS that output tiles are in.
        If not provided, tiles use the crs of `src` by default. Cannot be specified
        along with project_to_meters.
    project_to_meters : bool, optional
        Specifies whether to project to the correct utm zone for the location.
        Cannot be specified along with `dest_crs`.
    nodata : int, optional
        The value in `src` that specifies nodata. If this value is not
        provided, solaris will attempt to infer the nodata value from the `src`
        metadata.
    alpha : int, optional
        The band to specify as alpha. If not provided, solaris will attempt to
        infer if an alpha band is present from the `src` metadata.
    force_load_cog : bool, optional
        If `src` is a cloud-optimized geotiff, use this argument to force
        loading in the entire image at once.
    aoi_boundary : :class:`shapely.geometry.Polygon` or `list`-like [left, bottom, right, top]
        Defines the bounds of the AOI in which tiles will be created. If a
        tile will extend beyond the boundary, the "extra" pixels will have
        the value `nodata`. Can be provided at initialization of the :class:`Tiler`
        instance or when the input is loaded. If not provided either upon
        initialization or when an image is loaded, the image bounds will be
        used; if provided, this value will override image metadata.
    tile_bounds : `list`-like
        A `list`-like of ``[left, bottom, right, top]`` lists of coordinates
        defining the boundaries of the tiles to create. If not provided, they
        will be generated from the `aoi_boundary` based on `src_tile_size`.
    verbose : bool, optional
        Verbose text output. By default, verbose text is not printed.
    Attributes
    ----------
    src : :class:`rasterio.io.DatasetReader`
        The source dataset to tile.
    src_path : `str`
        The path or URL to the source dataset. Used for calling
        ``rio_cogeo.cogeo.cog_validate()``.
    dest_dir : `str`
        The directory to save the output tiles to. If not
    dest_crs : int
        The EPSG code for the output images. If not provided, outputs will
        keep the same CRS as the source image when ``Tiler.make_tile_images()``
        is called.
    tile_size: tuple
        A ``(y, x)`` :class:`tuple` storing the dimensions of the output.
        These are in pixel units unless ``size_in_meters=True``.
    size_in_meters : bool
        If ``True``, the units of `tile_size` are in meters instead of pixels.
    is_cog : bool
        Indicates whether or not the image being tiled is a Cloud-Optimized
        GeoTIFF (COG). Determined by checking COG validity using
        `rio-cogeo <https://github.com/cogeotiff/rio-cogeo>`_.
    nodata : `int`
        The value for nodata in the outputs. Will be set to zero in outputs if
        ``None``.
    alpha : `int`
        The band index corresponding to an alpha channel (if one exists).
        ``None`` if there is no alpha channel.
    tile_bounds : list
        A :class:`list` containing ``[left, bottom, right, top]`` bounds
        sublists for each tile created.
    resampling : str
        The resampling method for any resizing. Possible values are
        ``['bilinear', 'cubic', 'nearest', 'lanczos', 'average']`` (or any
        other option from :class:`rasterio.warp.Resampling`).
    aoi_boundary : :class:`shapely.geometry.Polygon`
        A :class:`shapely.geometry.Polygon` defining the bounds of the AOI that
        tiles will be created for. If a tile will extend beyond the boundary,
        the "extra" pixels will have the value `nodata`. Can be provided at
        initialization of the :class:`Tiler` instance or when the input is
        loaded.
    """

    def __init__(self, dest_dir=None, dest_crs=None, project_to_meters=False,
                 channel_idxs=None, src_tile_size=(900, 900), use_src_metric_size=False,
                 dest_tile_size=None, dest_metric_size=False,
                 aoi_boundary=None, nodata=None, alpha=None,
                 force_load_cog=False, resampling=None, tile_bounds=None, tile_overlay=128,
                 verbose=False):
        # set up attributes
        if verbose:
            print("Initializing Tiler...")
        self.dest_dir = dest_dir
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        if dest_crs is not None:
            self.dest_crs = _check_crs(dest_crs)
        else:
            self.dest_crs = None
        self.src_tile_size = src_tile_size
        self.use_src_metric_size = use_src_metric_size
        if dest_tile_size is None:
            self.dest_tile_size = src_tile_size
        else:
            self.dest_tile_size = dest_tile_size
        self.resampling = resampling
        self.force_load_cog = force_load_cog
        self.nodata = nodata
        self.alpha = alpha
        self.aoi_boundary = aoi_boundary
        self.tile_bounds = tile_bounds
        self.tile_overlay = tile_overlay
        self.project_to_meters = project_to_meters
        self.tile_paths = []  # retains the paths of the last call to .tile()
#        self.cog_output = cog_output
        self.verbose = verbose
        if self.verbose:
            print('Tiler initialized.')
            print('dest_dir: {}'.format(self.dest_dir))
            if dest_crs is not None:
                print('dest_crs: {}'.format(self.dest_crs))
            else:
                print('dest_crs will be inferred from source data.')
            print('src_tile_size: {}'.format(self.src_tile_size))
            print('tile size units metric: {}'.format(self.use_src_metric_size))
            if self.resampling is not None:
                print('Resampling is set to {}'.format(self.resampling))
            else:
                print('Resampling is set to None')

    def tile(self, src, dest_dir=None, channel_idxs=None, nodata=None,
             alpha=None, restrict_to_aoi=False,
             dest_fname_base=None, nodata_threshold = None):
        """An object to tile geospatial image strips into smaller pieces.
        Arguments
        ---------
        src : :class:`rasterio.io.DatasetReader` or str
            The source dataset to tile.
        nodata_threshold : float, optional
            Nodata percentages greater than this threshold will not be saved as tiles.
        restrict_to_aoi : bool, optional
            Requires aoi_boundary. Sets all pixel values outside the aoi_boundary to the nodata value of the src image.
        """
        src = _check_rasterio_im_load(src)
        restricted_im_path = os.path.join(self.dest_dir, "aoi_restricted_"+ os.path.basename(src.name))
        self.src_name = src.name # preserves original src name in case restrict is used
        if restrict_to_aoi is True:
            if self.aoi_boundary is None:
                raise ValueError("aoi_boundary must be specified when RasterTiler is called.")
            mask_geometry = self.aoi_boundary.intersection(box(*src.bounds)) # prevents enlarging raster to size of aoi_boundary
            index_lst = list(np.arange(1,src.meta['count']+1))
            # no need to use transform t since we don't crop. cropping messes up transform of tiled outputs
            arr, t = rasterio_mask(src, [mask_geometry], all_touched=False, invert=False, nodata=src.meta['nodata'],
                         filled=True, crop=False, pad=False, pad_width=0.5, indexes=list(index_lst))
            with rasterio.open(restricted_im_path, 'w', **src.profile) as dest:
                dest.write(arr)
                dest.close()
                src.close()
            src = _check_rasterio_im_load(restricted_im_path) #if restrict_to_aoi, we overwrite the src to be the masked raster

        tile_gen = self.tile_generator(src, dest_dir, channel_idxs, nodata,
                                       alpha, self.aoi_boundary, restrict_to_aoi)

        if self.verbose:
            print('Beginning tiling...')
        self.tile_paths = []
        if nodata_threshold is not None:
            if nodata_threshold > 1:
                raise ValueError("nodata_threshold should be expressed as a float less than 1.")
            print("nodata value threshold supplied, filtering based on this percentage.")
            new_tile_bounds = []
            for tile_data, mask, profile, tb in tqdm(tile_gen):
                nodata_count = np.logical_or.reduce((tile_data == profile['nodata']), axis=0).sum()
                nodata_perc = nodata_count / (tile_data.shape[1] * tile_data.shape[2])
                if nodata_perc < nodata_threshold:
                    dest_path = self.save_tile(
                        tile_data, mask, profile, dest_fname_base)
                    self.tile_paths.append(dest_path)
                    new_tile_bounds.append(tb)
                else:
                    print("{} of nodata is over the nodata_threshold, tile not saved.".format(nodata_perc))
            self.tile_bounds = new_tile_bounds # only keep the tile bounds that make it past the nodata threshold
        else:
            for tile_data, mask, profile, tb in tqdm(tile_gen):
                dest_path = self.save_tile(
                    tile_data, mask, profile, dest_fname_base)
                self.tile_paths.append(dest_path)
        if self.verbose:
            print('Tiling complete. Cleaning up...')
        self.src.close()
        if os.path.exists(os.path.join(self.dest_dir, 'tmp.tif')):
            os.remove(os.path.join(self.dest_dir, 'tmp.tif'))
        if os.path.exists(restricted_im_path):
            os.remove(restricted_im_path)
        if self.verbose:
            print("Done. CRS returned for vector tiling.")
        return _check_crs(profile['crs'])  # returns the crs to be used for vector tiling

    def tile_generator(self, src, dest_dir=None, channel_idxs=None,
                       nodata=None, alpha=None, aoi_boundary=None,
                       restrict_to_aoi=False):
        """Create the tiled output imagery from input tiles.
        Uses the arguments provided at initialization to generate output tiles.
        First, tile locations are generated based on `Tiler.tile_size` and
        `Tiler.size_in_meters` given the bounds of the input image.
        Arguments
        ---------
        src : `str` or :class:`Rasterio.DatasetReader`
            The source data to tile from. If this is a "classic"
            (non-cloud-optimized) GeoTIFF, the whole image will be loaded in;
            if it's cloud-optimized, only the required portions will be loaded
            during tiling unless ``force_load_cog=True`` was specified upon
            initialization.
        dest_dir : str, optional
            The path to the destination directory to output images to. If the
            path doesn't exist, it will be created. This argument is required
            if it wasn't provided during initialization.
        channel_idxs : list, optional
            The list of channel indices to be included in the output array.
            If not provided, all channels will be included. *Note:* per
            ``rasterio`` convention, indexing starts at ``1``, not ``0``.
        nodata : int, optional
            The value in `src` that specifies nodata. If this value is not
            provided, solaris will attempt to infer the nodata value from the
            `src` metadata.
        alpha : int, optional
            The band to specify as alpha. If not provided, solaris will attempt
            to infer if an alpha band is present from the `src` metadata.
        aoi_boundary : `list`-like or :class:`shapely.geometry.Polygon`, optional
            AOI bounds can be provided either as a
            ``[left, bottom, right, top]`` :class:`list`-like or as a
            :class:`shapely.geometry.Polygon`.
        restrict_to_aoi : bool, optional
            Should output tiles be restricted to the limits of the AOI? If
            ``True``, any tile that partially extends beyond the limits of the
            AOI will not be returned. This is the inverse of the ``boundless``
            argument for :class:`rasterio.io.DatasetReader` 's ``.read()``
            method.
        Yields
        ------
        tile_data, mask, tile_bounds
            tile_data : :class:`numpy.ndarray`
            A list of lists of each tile's bounds in the order they were
            created, to be used in tiling vector data. These data are also
            stored as an attribute of the :class:`Tiler` instance named
            `tile_bounds`.
        """
        # parse arguments
        if self.verbose:
            print("Checking input data...")
        # if isinstance(src, str):
        #     self.is_cog = cog_validate(src)
        # else:
        # self.is_cog = cog_validate(src.name)
        # if self.verbose:
        #     print('COG: {}'.format(self.is_cog))
        self.src = _check_rasterio_im_load(src)
        if channel_idxs is None:  # if not provided, include them all
            channel_idxs = list(range(1, self.src.count + 1))
            print(channel_idxs)
        self.src_crs = _check_crs(self.src.crs, return_rasterio=True) # necessary to use rasterio crs for reproject
        if self.verbose:
            print('Source CRS: EPSG:{}'.format(self.src_crs.to_epsg()))
        if self.dest_crs is None:
            self.dest_crs = self.src_crs
        if self.verbose:
            print('Destination CRS: EPSG:{}'.format(self.dest_crs.to_epsg()))
        self.src_path = self.src.name
        self.proj_unit = raster_get_projection_unit(self.src)  # for rounding
        if self.verbose:
            print("Inputs OK.")
        if self.use_src_metric_size:
            if self.verbose:
                print("Checking if inputs are in metric units...")
            if self.project_to_meters:
                if self.verbose:
                    print("Input CRS is not metric. "
                          "Reprojecting the input to UTM.")
                self.src = reproject(self.src,
                                     resampling_method=self.resampling,
                                     dest_path=os.path.join(self.dest_dir,
                                                            'tmp.tif'))
                if self.verbose:
                    print('Done reprojecting.')
        if nodata is None and self.nodata is None:
            self.nodata = self.src.nodata
        elif nodata is not None:
            self.nodata = nodata
        # get index of alpha channel
        if alpha is None and self.alpha is None:
            mf_list = [rasterio.enums.MaskFlags.alpha in i for i in
                       self.src.mask_flag_enums]  # list with True at idx of alpha c
            try:
                self.alpha = np.where(mf_list)[0] + 1
            except IndexError:  # if there isn't a True
                self.alpha = None
        else:
            self.alpha = alpha

        if getattr(self, 'tile_bounds', None) is None:
            self.get_tile_bounds()

        for tb in self.tile_bounds:
            # removing the following line until COG functionality implemented
            if True:  # not self.is_cog or self.force_load_cog:
                window = rasterio.windows.from_bounds(
                    *tb, transform=self.src.transform,
                    width=self.src_tile_size[1],
                    height=self.src_tile_size[0])
                if self.src.count != 1:
                    src_data = self.src.read(
                        window=window,
                        indexes=channel_idxs,
                        boundless=True,
                        fill_value=self.nodata)
                else:
                    src_data = self.src.read(
                        window=window,
                        boundless=True,
                        fill_value=self.nodata)

                dst_transform, width, height = calculate_default_transform(
                    self.src.crs, self.dest_crs,
                    self.src.width, self.src.height, *tb,
                    dst_height=self.dest_tile_size[0],
                    dst_width=self.dest_tile_size[1])

                if self.dest_crs != self.src_crs and self.resampling_method is not None:
                    tile_data = np.zeros(shape=(src_data.shape[0], height, width), dtype=src_data.dtype)
                    rasterio.warp.reproject(
                        source=src_data,
                        destination=tile_data,
                        src_transform=self.src.window_transform(window),
                        src_crs=self.src.crs,
                        dst_transform=dst_transform,
                        dst_crs=self.dest_crs,
                        dst_nodata=self.nodata,
                        resampling=getattr(Resampling, self.resampling))

                elif self.dest_crs != self.src_crs and self.resampling_method is None:
                    print("Warning: You've set resampling to None but your "
                          "destination projection differs from the source "
                          "projection. Using bilinear resampling by default.")
                    tile_data = np.zeros(shape=(src_data.shape[0], height, width),
                                         dtype=src_data.dtype)
                    tile_data = np.zeros(shape=(src_data.shape[0], height, width), dtype=src_data.dtype)
                    rasterio.warp.reproject(
                        source=src_data,
                        destination=tile_data,
                        src_transform=self.src.window_transform(window),
                        src_crs=self.src.crs,
                        dst_transform=dst_transform,
                        dst_crs=self.dest_crs,
                        dst_nodata=self.nodata,
                        resampling=getattr(Resampling, "bilinear"))

                else:  # for the case where there is no resampling and no dest_crs specified, no need to reproject or resample

                    tile_data = src_data

                if self.nodata:
                    mask = np.all(tile_data != nodata,
                                  axis=0).astype(np.uint8) * 255
                elif self.alpha:
                    mask = self.src.read(self.alpha, window=window)
                else:
                    mask = None  # placeholder

            # else:
            #     tile_data, mask, window, aff_xform = read_cog_tile(
            #         src=self.src,
            #         bounds=tb,
            #         tile_size=self.dest_tile_size,
            #         indexes=channel_idxs,
            #         nodata=self.nodata,
            #         resampling_method=self.resampling
            #         )
            profile = self.src.profile
            profile.update(width=self.dest_tile_size[1],
                           height=self.dest_tile_size[0],
                           crs=self.dest_crs,
                           transform=dst_transform)
            if len(tile_data.shape) == 2:  # if there's no channel band
                profile.update(count=1)
            else:
                profile.update(count=tile_data.shape[0])

            yield tile_data, mask, profile, tb

    def save_tile(self, tile_data, mask, profile, dest_fname_base=None):
        """Save a tile created by ``Tiler.tile_generator()``."""
        if dest_fname_base is None:
            dest_fname_root = os.path.splitext(
                os.path.split(self.src_path)[1])[0]
        else:
            dest_fname_root = dest_fname_base
        if self.proj_unit not in ['meter', 'metre']:
            dest_fname = '{}_{}_{}.tif'.format(
                dest_fname_root,
                np.round(profile['transform'][2], 3),
                np.round(profile['transform'][5], 3))
        else:
            dest_fname = '{}_{}_{}.tif'.format(
                dest_fname_root,
                int(profile['transform'][2]),
                int(profile['transform'][5]))
        # if self.cog_output:
        #     dest_path = os.path.join(self.dest_dir, 'tmp.tif')
        # else:
        dest_path = os.path.join(self.dest_dir, dest_fname)

        with rasterio.open(dest_path, 'w',
                           **profile) as dest:
            if profile['count'] == 1:
                dest.write(tile_data[0, :, :], 1)
            else:
                for band in range(1, profile['count'] + 1):
                    # base-1 vs. base-0 indexing...bleh
                    dest.write(tile_data[band-1, :, :], band)
            if self.alpha:
                # write the mask if there's an alpha band
                dest.write(mask, profile['count'] + 1)

            dest.close()

        return dest_path

        # if self.cog_output:
        #     self._create_cog(os.path.join(self.dest_dir, 'tmp.tif'),
        #                      os.path.join(self.dest_dir, dest_fname))
        #     os.remove(os.path.join(self.dest_dir, 'tmp.tif'))

    def fill_all_nodata(self, nodata_fill):
        """
        Fills all tile nodata values with a fill value.
        The standard workflow is to run this function only after generating label masks and using the original output
        from the raster tiler to filter out label pixels that overlap nodata pixels in a tile. For example,
        solaris.vector.mask.instance_mask will filter out nodata pixels from a label mask if a reference_im is provided,
        and after this step nodata pixels may be filled by calling this method.
        nodata_fill : int, float, or str, optional
            Default is to not fill any nodata values. Otherwise, pixels outside of the aoi_boundary and pixels inside
            the aoi_boundary with the nodata value will be filled. "mean" will fill pixels with the channel-wise mean.
            Providing an int or float will fill pixels in all channels with the provided value.
        Returns: list
            The fill values, in case the mean of the src image should be used for normalization later.
        """
        src = _check_rasterio_im_load(self.src_name)
        if nodata_fill == "mean":
            arr = src.read()
            arr_nan = np.where(arr!=src.nodata, arr, np.nan)
            fill_values = np.nanmean(arr_nan, axis=tuple(range(1, arr_nan.ndim)))
            print('Fill values set to {}'.format(fill_values))
        elif isinstance(nodata_fill, (float, int)):
            fill_values = src.meta['count'] * [nodata_fill]
            print('Fill values set to {}'.format(fill_values))
        else:
            raise TypeError('nodata_fill must be "mean", int, or float. {} was supplied.'.format(nodata_fill))
        src.close()
        for tile_path in self.tile_paths:
            tile_src = rasterio.open(tile_path, "r+")
            tile_data = tile_src.read()
            for i in np.arange(tile_data.shape[0]):
                tile_data[i,...][tile_data[i,...] == tile_src.nodata] = fill_values[i] # set fill value for each band
            if tile_src.meta['count'] == 1:
                tile_src.write(tile_data[0, :, :], 1)
            else:
                for band in range(1, tile_src.meta['count'] + 1):
                    # base-1 vs. base-0 indexing...bleh
                    tile_src.write(tile_data[band-1, :, :], band)
            tile_src.close()
        return fill_values

    def _create_cog(self, src_path, dest_path):
        """Overwrite non-cloud-optimized GeoTIFF with a COG."""
        cog_translate(src_path=src_path, dst_path=dest_path,
                      dst_kwargs={'crs': self.dest_crs},
                      resampling=self.resampling,
                      latitude_adjustment=False)

    def get_tile_bounds(self):
        """Get tile bounds for each tile to be created in the input CRS."""
        if not self.aoi_boundary:
            if not self.src:
                raise ValueError('aoi_boundary and/or a source file must be '
                                 'provided.')
            else:
                # set to the bounds of the image
                # split_geom can take a list
                self.aoi_boundary = list(self.src.bounds)

        self.tile_bounds = split_geom(geometry=self.aoi_boundary, tile_size=self.src_tile_size, tile_overlay = self.tile_overlay, resolution=(
            self.src.transform[0], -self.src.transform[4]), use_projection_units=self.use_src_metric_size, src_img=self.src)

    def load_src_vrt(self):
        """Load a source dataset's VRT into the destination CRS."""
        vrt_params = dict(crs=self.dest_crs,
                          resampling=getattr(Resampling, self.resampling),
                          src_nodata=self.nodata, dst_nodata=self.nodata)
        return WarpedVRT(self.src, **vrt_params)
    

    
class VectorTiler(object):
    """An object to tile geospatial vector data into smaller pieces.
    Arguments
    ---------
    Attributes
    ----------
    """

    def __init__(self, dest_dir=None, dest_crs=None, output_format='GeoJSON',
                 verbose=False, super_verbose=False):
        if verbose or super_verbose:
            print('Preparing the tiler...')
        self.dest_dir = dest_dir
        if not os.path.isdir(self.dest_dir):
            os.makedirs(self.dest_dir)
        if dest_crs is not None:
            self.dest_crs = _check_crs(dest_crs)
        self.output_format = output_format
        self.verbose = verbose
        self.super_verbose = super_verbose
        self.tile_paths = [] # retains the paths of the last call to .tile()
        if self.verbose or self.super_verbose:
            print('Initialization done.')

    def tile(self, src, tile_bounds, tile_bounds_crs=None, geom_type='Polygon',
             split_multi_geoms=True, min_partial_perc=0.0,
             dest_fname_base='geoms', obj_id_col=None,
             output_ext='.geojson'):
        """Tile `src` into vector data tiles bounded by `tile_bounds`.
        Arguments
        ---------
        src : `str` or :class:`geopandas.GeoDataFrame`
            The source vector data to tile. Must either be a path to a GeoJSON
            or a :class:`geopandas.GeoDataFrame`.
        tile_bounds : list
            A :class:`list` made up of ``[left, top, right, bottom] `` sublists
            (this can be extracted from
            :class:`solaris.tile.raster_tile.RasterTiler` after tiling imagery)
        tile_bounds_crs : int, optional
            The EPSG code or rasterio.crs.CRS object for the CRS that the tile
            bounds are in. RasterTiler.tile returns the CRS of the raster tiles
            and can be used here. If not provided, it's assumed that the CRS is the
            same as in `src`. This argument must be provided if the bound
            coordinates and `src` are not in the same CRS, otherwise tiling will
            not occur correctly.
        geom_type : str, optional (default: "Polygon")
            The type of geometries contained within `src`. Defaults to
            ``"Polygon"``, can also be ``"LineString"``.
        split_multi_geoms : bool, optional (default: True)
            Should multi-polygons or multi-linestrings generated by clipping
            a geometry into discontinuous pieces be separated? Defaults to yes
            (``True``).
        min_partial_perc : float, optional (default: 0.0)
            The minimum percentage of a :class:`shapely.geometry.Polygon` 's
            area or :class:`shapely.geometry.LineString` 's length that must
            be retained within a tile's bounds to be included in the output.
            Defaults to ``0.0``, meaning that the contained portion of a
            clipped geometry will be included, no matter how small.
        dest_fname_base : str, optional (default: 'geoms')
            The base filename to use when creating outputs. The lower left
            corner coordinates of the tile's bounding box will be appended
            when saving.
        obj_id_col : str, optional (default: None)
            If ``split_multi_geoms=True``, the name of a column that specifies
            a unique identifier for each geometry (e.g. the ``"BuildingId"``
            column in many SpaceNet datasets.) See
            :func:`solaris.utils.geo.split_multi_geometries` for more.
        output_ext : str, optional, (default: geojson)
            Extension of output files, can be 'geojson' or 'json'.
        """

        if isinstance(src, gpd.GeoDataFrame) and src.crs is None:
            raise ValueError("If the src input is a geopandas.GeoDataFrame, it must have a crs attribute.")

        tile_gen = self.tile_generator(src, tile_bounds, tile_bounds_crs,
                                       geom_type, split_multi_geoms,
                                       min_partial_perc,
                                       obj_id_col=obj_id_col)
        self.tile_paths = []
        for tile_gdf, tb in tqdm(tile_gen):
            if self.proj_unit not in ['meter', 'metre']:
                dest_path = os.path.join(
                    self.dest_dir, '{}_{}_{}{}'.format(dest_fname_base,
                                                       np.round(tb[0], 3),
                                                       np.round(tb[3], 3),
                                                       output_ext))
            else:
                dest_path = os.path.join(
                    self.dest_dir, '{}_{}_{}{}'.format(dest_fname_base,
                                                       int(tb[0]),
                                                       int(tb[3]),
                                                       output_ext))
            self.tile_paths.append(dest_path)
            if len(tile_gdf) > 0:
                tile_gdf.to_file(dest_path, driver='GeoJSON')
            else:
                save_empty_geojson(dest_path, self.dest_crs)

    def tile_generator(self, src, tile_bounds, tile_bounds_crs=None,
                       geom_type='Polygon', split_multi_geoms=True,
                       min_partial_perc=0.0, obj_id_col=None):
        """Generate `src` vector data tiles bounded by `tile_bounds`.
        Arguments
        ---------
        src : `str` or :class:`geopandas.GeoDataFrame`
            The source vector data to tile. Must either be a path to a GeoJSON
            or a :class:`geopandas.GeoDataFrame`.
        tile_bounds : list
            A :class:`list` made up of ``[left, top, right, bottom] `` sublists
            (this can be extracted from
            :class:`solaris.tile.raster_tile.RasterTiler` after tiling imagery)
        tile_bounds_crs : int, optional
            The EPSG code for the CRS that the tile bounds are in. If not
            provided, it's assumed that the CRS is the same as in `src`. This
            argument must be provided if the bound coordinates and `src` are
            not in the same CRS, otherwise tiling will not occur correctly.
        geom_type : str, optional (default: "Polygon")
            The type of geometries contained within `src`. Defaults to
            ``"Polygon"``, can also be ``"LineString"``.
        split_multi_geoms : bool, optional (default: True)
            Should multi-polygons or multi-linestrings generated by clipping
            a geometry into discontinuous pieces be separated? Defaults to yes
            (``True``).
        min_partial_perc : float, optional (default: 0.0)
            The minimum percentage of a :class:`shapely.geometry.Polygon` 's
            area or :class:`shapely.geometry.LineString` 's length that must
            be retained within a tile's bounds to be included in the output.
            Defaults to ``0.0``, meaning that the contained portion of a
            clipped geometry will be included, no matter how small.
        obj_id_col : str, optional (default: None)
            If ``split_multi_geoms=True``, the name of a column that specifies
            a unique identifier for each geometry (e.g. the ``"BuildingId"``
            column in many SpaceNet datasets.) See
            :func:`solaris.utils.geo.split_multi_geometries` for more.
        Yields
        ------
        tile_gdf : :class:`geopandas.GeoDataFrame`
            A tile geodataframe.
        tb : list
            A list with ``[left, top, right, bottom] `` coordinates for the
            boundaries contained by `tile_gdf`.
        """
        self.src = _check_gdf_load(src)
        if self.verbose:
            print("Num tiles:", len(tile_bounds))

        self.src_crs = _check_crs(self.src.crs)
        # check if the tile bounds and vector are in the same crs
        if tile_bounds_crs is not None:
            tile_bounds_crs = _check_crs(tile_bounds_crs)
        else:
            tile_bounds_crs = self.src_crs
        if self.src_crs != tile_bounds_crs:
            reproject_bounds = True  # used to transform tb for clip_gdf()
        else:
            reproject_bounds = False

        self.proj_unit = get_projection_unit(self.src_crs)
        if getattr(self, 'dest_crs', None) is None:
            self.dest_crs = self.src_crs
        for i, tb in enumerate(tile_bounds):
            if self.super_verbose:
                print("\n", i, "/", len(tile_bounds))
            if reproject_bounds:
                tile_gdf = clip_gdf(self.src,
                                    reproject_geometry(box(*tb),
                                                       tile_bounds_crs,
                                                       self.src_crs),
                                    min_partial_perc,
                                    geom_type, verbose=self.super_verbose)
            else:
                tile_gdf = clip_gdf(self.src, tb, min_partial_perc, geom_type,
                                    verbose=self.super_verbose)
            if self.src_crs != self.dest_crs:
                tile_gdf = tile_gdf.to_crs(crs=self.dest_crs.to_wkt())
            if split_multi_geoms:
                split_multi_geometries(tile_gdf, obj_id_col=obj_id_col)
            yield tile_gdf, tb


def search_gdf_polygon(gdf, tile_polygon):
    """Find polygons in a GeoDataFrame that overlap with `tile_polygon` .
    Arguments
    ---------
    gdf : :py:class:`geopandas.GeoDataFrame`
        A :py:class:`geopandas.GeoDataFrame` of polygons to search.
    tile_polygon : :py:class:`shapely.geometry.Polygon`
        A :py:class:`shapely.geometry.Polygon` denoting a tile's bounds.
    Returns
    -------
    precise_matches : :py:class:`geopandas.GeoDataFrame`
        The subset of `gdf` that overlaps with `tile_polygon` . If
        there are no overlaps, this will return an empty
        :py:class:`geopandas.GeoDataFrame`.
    """
    sindex = gdf.sindex
    possible_matches_index = list(sindex.intersection(tile_polygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[
        possible_matches.intersects(tile_polygon)
        ]
    if precise_matches.empty:
        precise_matches = gpd.GeoDataFrame(geometry=[])
    return precise_matches


def clip_gdf(gdf, tile_bounds, min_partial_perc=0.0, geom_type="Polygon",
             use_sindex=True, verbose=False):
    """Clip GDF to a provided polygon.
    Clips objects within `gdf` to the region defined by
    `poly_to_cut`. Also adds several columns to the output::
        `origarea`
            The original area of the polygons (only used if `geom_type` ==
            ``"Polygon"``).
        `origlen`
            The original length of the objects (only used if `geom_type` ==
            ``"LineString"``).
        `partialDec`
            The fraction of the object that remains after clipping
            (fraction of area for Polygons, fraction of length for
            LineStrings.) Can filter based on this by using `min_partial_perc`.
        `truncated`
            Boolean indicator of whether or not an object was clipped.
    Arguments
    ---------
    gdf : :py:class:`geopandas.GeoDataFrame`
        A :py:class:`geopandas.GeoDataFrame` of polygons to clip.
    tile_bounds : `list` or :class:`shapely.geometry.Polygon`
        The geometry to clip objects in `gdf` to. This can either be a
        ``[left, top, right, bottom] `` bounds list or a
        :class:`shapely.geometry.Polygon` object defining the area to keep.
    min_partial_perc : float, optional
        The minimum fraction of an object in `gdf` that must be
        preserved. Defaults to 0.0 (include any object if any part remains
        following clipping).
    geom_type : str, optional
        Type of objects in `gdf`. Can be one of
        ``["Polygon", "LineString"]`` . Defaults to ``"Polygon"`` .
    use_sindex : bool, optional
        Use the `gdf` sindex be used for searching. Improves efficiency
        but requires `libspatialindex <http://libspatialindex.github.io/>`__ .
    verbose : bool, optional
        Switch to print relevant values.
    Returns
    -------
    cut_gdf : :py:class:`geopandas.GeoDataFrame`
        `gdf` with all contained objects clipped to `poly_to_cut` .
        See notes above for details on additional clipping columns added.
    """
    if isinstance(tile_bounds, tuple):
        tb = box(*tile_bounds)
    elif isinstance(tile_bounds, list):
        tb = box(*tile_bounds)
    elif isinstance(tile_bounds, Polygon):
        tb = tile_bounds
    if use_sindex and (geom_type == "Polygon"):
        gdf = search_gdf_polygon(gdf, tb)

    # if geom_type == "LineString":
    if 'origarea' in gdf.columns:
        pass
    else:
        if "geom_type" == "LineString":
            gdf['origarea'] = 0
        else:
            gdf['origarea'] = gdf.area

    if 'origlen' in gdf.columns:
        pass
    else:
        if "geom_type" == "LineString":
            gdf['origlen'] = gdf.length
        else:
            gdf['origlen'] = 0
    # TODO must implement different case for lines and for spatialIndex
    # (Assume RTree is already performed)

    cut_gdf = gdf.copy()
    cut_gdf.geometry = gdf.intersection(tb)

    if geom_type == 'Polygon':
        cut_gdf['partialDec'] = cut_gdf.area / cut_gdf['origarea']
        cut_gdf = cut_gdf.loc[cut_gdf['partialDec'] > min_partial_perc, :]
        cut_gdf['truncated'] = (cut_gdf['partialDec'] != 1.0).astype(int)
    else:
        # assume linestrings
        # remove null
        cut_gdf = cut_gdf[cut_gdf['geometry'].notnull()]
        cut_gdf['partialDec'] = 1
        cut_gdf['truncated'] = 0
        # cut_gdf = cut_gdf[cut_gdf.geom_type != "GeometryCollection"]
        if len(cut_gdf) > 0 and verbose:
            print("clip_gdf() - gdf.iloc[0]:", gdf.iloc[0])
            print("clip_gdf() - tb:", tb)
            print("clip_gdf() - gdf_cut:", cut_gdf)

    # TODO: IMPLEMENT TRUNCATION MEASUREMENT FOR LINESTRINGS

    return cut_gdf