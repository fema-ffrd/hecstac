# extension

## Overview

Contains JSON schema used for defining and validating terms relating to properties extracted for 1D HEC RAS models using a project PRJ file

## Item properties

Item properties include:

- has_2d
- has_1d
- datetime_source
- array of items documenting linked USGS gages with following properties:
  - usgs_gage_name
  - usgs_gage_site_number
  - usgs_gage_point
  - associated_model_reference_line_name

## Asset properties

(** indicates that a property is referenced in different assets, ++ indicates that it has been identified as ignorable)

Possible asset properties for project assets include:

- project_title
- ras_version
- model_name
- ras_version
- project_units

Possible properties for links between assets:

- link_type (a string value which describes the specific way in which an asset is related to another asset)

  - possible values include [plan_referenced, plan_current, geometry_referenced, steady_flow_referenced, quasi_unsteady_flow_referenced, unsteady_flow_referenced, landcover_referenced, terrain_referenced, meteorology_dss_referenced]

Possible asset properties for plan assets include:

- plan_title **
- primary_geometry
- primary_flow
- short_identifier **

Possible asset properties for geom assets include:

- geom_title **
- rivers
- reaches
- cross_sections
- culverts **?
- bridges **?
- multiple_openings
- inline_structures **
- lateral_structures **
- storage_areas
- 2d_flow_areas
- sa_connections

Possible asset properties for steady flow assets include:

- flow_title **
- number_of_profiles

Possible asset properties for quasi-unsteady flow assets include:

- flow_title **

Possible asset properties for unsteady flow assets include:

- flow_title **

Possible asset properties for HDF assets include:

- file_version
- units_system
- complete_geometry ++
- geometry_time
- geometry_land_cover_date_last_modified
- geometry_land_cover_filename
- geometry_land_cover_layername
- geometry_rasmapperlibdll_date
- geometry_si_units
- geometry_terrain_file_date
- geometry_terrain_filename
- geometry_terrain_layername
- geometry_version
- structures_bridgeculvert_count
- structures_connection_count
- structures_inline_structure_count
- structures_lateral_structure_count
- 2d_flow_cell_average_size
- 2d_flow_cell_maximum_index
- 2d_flow_cell_maximum_size
- 2d_flow_cell_minimum_size
- 2d_flow_cell_volume_tolerance ++
- 2d_flow_data_date ++
- 2d_flow_extents ++
- 2d_flow_face_area_conveyance_ratio ++
- 2d_flow_face_area_elevation_tolerance ++
- 2d_flow_face_profile_tolerance ++
- 2d_flow_areas_infiltration_file_date ++
- 2d_flow_areas_infiltration_filename ++
- 2d_flow_areas_infiltration_layername ++
- 2d_flow_infiltration_override_table_hash ++
- 2d_flow_land_cover_date_last_modified ++
- 2d_flow_land_cover_file_date ++
- 2d_flow_land_cover_filename ++
- 2d_flow_land_cover_layername ++
- 2d_flow_mannings_n ++
- 2d_flow_areas_property_tables_last_computed ++
- 2d_flow_areas:terrain_file_date ++
- 2d_flow_areas:terrain_filename ++
- 2d_flow_property_tables_lc_hash ++
- 2d_flow_version ++

Asset properties for plan HDF assets include:

- plan_information_base_output_interval
- plan_information_computation_time_step_base
- plan_information_flow_filename
- plan_information_flow_title ++ **
- plan_information_geometry_filename
- plan_information_geometry_title ++ **
- plan_information_plan_filename
- plan_information_plan_name
- plan_information_plan_shortid ++ **
- plan_information_plan_title ++ **
- plan_information_project_filename
- plan_information_project_title
- plan_information_simulation_end_time
- plan_information_simulation_start_time
- plan_information_time_window ++
- plan_parameters_1d_cores ++
- plan_parameters_1d_flow_tolerance
- plan_parameters_1d_maximum_iterations
- plan_parameters_1d_maximum_iterations_without_improvement
- plan_parameters_1d_maximum_water_surface_error_to_abort
- plan_parameters_1d_methodology ++
- plan_parameters_1d_storage_area_elevation_tolerance
- plan_parameters_1d_theta
- plan_parameters_1d_theta_warmup
- plan_parameters_1d_water_surface_elevation_tolerance
- plan_parameters_1d2d_flow_tolerance ++
- plan_parameters_1d2d_gate_flow_submergence_decay_exponent
- plan_parameters_1d2d_is_stablity_factor
- plan_parameters_1d2d_ls_stablity_factor
- plan_parameters_1d2d_maximum_iterations ++
- plan_parameters_1d2d_maximum_number_of_time_slices
- plan_parameters_1d2d_minimum_flow_tolerance ++
- plan_parameters_1d2d_minimum_time_step_for_slicinghours
- plan_parameters_1d2d_number_of_warmup_steps
- plan_parameters_1d2d_warmup_time_step_hours
- plan_parameters_1d2d_water_surface_tolerance ++
- plan_parameters_1d2d_weir_flow_submergence_decay_exponent
- plan_parameters_1d2d_maxiter
- plan_parameters_1d2d_ws_tolerance ++
- plan_parameters_2d_boundary_condition_ramp_up_fraction ++
- plan_parameters_2d_boundary_condition_volume_check ++
- plan_parameters_2d_cores_per_mesh ++
- plan_parameters_2d_coriolis ++
- plan_parameters_2d_equation_set
- plan_parameters_2d_initial_conditions_ramp_up_time_hrs ++
- plan_parameters_2d_latitude_for_coriolis ++
- plan_parameters_2d_longitudinal_mixing_coefficient ++
- plan_parameters_2d_matrix_solver ++
- plan_parameters_2d_maximum_iterations ++
- plan_parameters_2d_names
- plan_parameters_2d_number_of_time_slices ++
- plan_parameters_2d_only ++
- plan_parameters_2d_smagorinsky_mixing_coefficient ++
- plan_parameters_2d_theta ++
- plan_parameters_2d_theta_warmup ++
- plan_parameters_2d_transverse_mixing_coefficient ++
- plan_parameters_2d_turbulence_formulation ++
- plan_parameters_2d_volume_tolerance
- plan_parameters_2d_water_surface_tolerance
- plan_parameters_gravity ++
- plan_parameters_hdf_chunk_size ++
- plan_parameters_hdf_compression ++
- plan_parameters_hdf_fixed_rows ++
- plan_parameters_hdf_flush_buffer ++
- plan_parameters_hdf_spatial_parts ++
- plan_parameters_hdf_use_max_rows ++
- plan_parameters_hdf_write_time_slices ++
- plan_parameters_hdf_write_warmup ++
- plan_parameters_pardiso_solver ++
- meteorology_dss_filename
- meteorology_dss_pathname
- meteorology_data_type
- meteorology_enabled ++
- meteorology_mode
- meteorology_raster_cellsize
- meteorology_raster_cols ++
- meteorology_raster_left ++
- meteorology_raster_rows ++
- meteorology_raster_top ++
- meteorology_source
- meteorology_units
- results_summary_computation_time_total_minutes ++
- unsteady_results_short_id ++
- volume_accounting_total_boundary_flux_of_water_in ++
- volume_accounting_total_boundary_flux_of_water_out ++
- volume_accounting_vol_accounting_in ++
- volume_accounting_error ++
- volume_accounting_volume_ending ++
- volume_accounting_volume_starting ++
- volume_accounting_precipitation_excess_acre_feet ++
- start_datetime ++
- end_datetime ++
- datetime ++
