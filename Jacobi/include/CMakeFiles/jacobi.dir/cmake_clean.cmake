file(REMOVE_RECURSE
  "libjacobi.pdb"
  "libjacobi.dylib"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/jacobi.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
