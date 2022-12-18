add_subdirectory(ops)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/params.json" "${CMAKE_CURRENT_SOURCE_DIR}/congestion.pth")
install(
    FILES ${INSTALL_SRCS} DESTINATION dreamplace
    )
