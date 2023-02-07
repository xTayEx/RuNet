ExternalProject_Add(
        png++
        PREFIX "png++"
        URL http://download.savannah.nongnu.org/releases/pngpp/png++-0.2.9.tar.gz
        DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/third_party/png++/download
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/png++/src
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
)
ExternalProject_Get_Property(png++ source_dir)
set(PNGPP_INCLUDE_DIR ${source_dir})
