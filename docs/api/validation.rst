Validation Package
==================

The validation package provides comprehensive data validation with helpful error messages.

.. automodule:: luxar.validation
   :members:
   :undoc-members:
   :show-inheritance:

Type Validation
---------------

Basic type validation and type guards for array data.

.. automodule:: luxar.validation.types
   :members:
   :undoc-members:

Write Validation
----------------

Detailed validation for data being written to Zarr, with helpful error messages.

.. automodule:: luxar.validation.base
   :members:
   :undoc-members:

nD Validation
-------------

Validation for n-dimensional data coverage and slicing.

.. automodule:: luxar.validation.nd
   :members:
   :undoc-members:

Category Validation
-------------------

Validation for categorical dimensions, extracted to avoid circular imports.

.. automodule:: luxar.validation.category_validation
   :members:
   :undoc-members:

   This module provides validation for category lists used in categorical dimensions.
   Categories must be non-empty lists of unique strings with reasonable length limits.
   
   Example::
   
      from luxar.validation.category_validation import validate_categories
      
      # Valid categories
      categories = validate_categories(["DAPI", "GFP", "mCherry"])
      
      # Raises ValueError for duplicates
      try:
          validate_categories(["A", "A", "B"])
      except ValueError as e:
          print(e)  # "Duplicate category names found: A"
