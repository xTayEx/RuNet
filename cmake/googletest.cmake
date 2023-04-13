include(FetchContent)

set(GOOGLETEST_GIT_URL https://github.com/google/googletest.git)
set(GOOGLETEST_GIT_TAG main)

FETCHCONTENT_DECLARE(
        googletest
        GIT_REPOSITORY ${GOOGLETEST_GIT_URL}
        GIT_TAG        ${GOOGLETEST_GIT_TAG}
        GIT_PROGRESS   TRUE
)

FETCHCONTENT_MAKEAVAILABLE(googletest)