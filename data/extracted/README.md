# Extracted Numerical Data From Provided Papers

These CSV files contain numerical values explicitly visible in `paper1.pdf` and `paper2.pdf` after text extraction. They are intended as provenance-tracked reference data for building a better prototype.

Important limitations:

- The complete raw training databases used by the papers are not present in the PDFs.
- `paper2.pdf` states that processed data required to reproduce the findings cannot be shared for legal or ethical reasons.
- Figure-only data were not digitized, because digitization from plots would create approximate values rather than validated raw data.
- Some tables are fragmented by PDF text extraction; ambiguous values are marked with `NA` or explained in the `notes` column.

Files:

- `paper1_search_space_bounds.csv`: Table 3 search-space bounds from paper1.
- `paper1_selected_alloys_table4_composition_properties.csv`: Table 4 nominal compositions, normalized costs, misfit and BTR from paper1.
- `paper1_table5_equilibrium_constitution_750C.csv`: Table 5 Thermo-Calc equilibrium constitution at 750 °C from paper1.
- `paper1_table6_predicted_properties_750C.csv`: Table 6 predicted/measured comparison properties at 750 °C from paper1.
- `paper2_table1_creep_database_variable_ranges.csv`: Table 1 variable ranges for the creep rupture stress database from paper2.
- `paper2_table2_lattice_misfit_coefficients.csv`: Table 2 Vegard coefficients for lattice-misfit calculation from paper2.
- `paper2_table3_density_model_coefficients.csv`: Table 3 density model coefficients from paper2.
- `paper2_table4_search_space_bounds.csv`: Table 4 grid-search bounds from paper2.
- `paper2_screening_summary.csv`: Numerical screening counts and thresholds reported in paper2 text.

Use these to replace some prototype placeholders carefully. For example, `paper2_table3_density_model_coefficients.csv` can replace the current simplistic density mixture rule, but the creep-strength model still needs the original trained Gaussian-process model or an independently rebuilt dataset.
