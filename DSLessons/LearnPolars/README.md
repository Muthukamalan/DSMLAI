# Polars Overview
- Polars is a Rust library with Python bindings
- Polars has expressions and contexts
    - contexts `.select`, `.with_columns`, `.filter`
    - expression -done w/ `pl.col("colum_name")` or `pl.lit(1)`
- lazy eval
- query optimization
- multi-threaded