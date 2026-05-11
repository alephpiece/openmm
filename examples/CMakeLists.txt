# Generate and install examples.

ADD_SUBDIRECTORY(cpp-examples)
ADD_SUBDIRECTORY(python-examples)
INSTALL(DIRECTORY benchmarks DESTINATION examples
    PATTERN log EXCLUDE
    PATTERN cache EXCLUDE
    PATTERN save-temps EXCLUDE
    PATTERN "*.db" EXCLUDE
    PATTERN "*.pftrace" EXCLUDE
)
INSTALL(DIRECTORY extras DESTINATION examples)
