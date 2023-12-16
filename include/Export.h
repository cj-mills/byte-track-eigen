#ifdef BUILDING_BYTE_TRACK_EIGEN
#ifdef _WIN32
#define BYTE_TRACK_EIGEN_API __declspec(dllexport)
#else
#define BYTE_TRACK_EIGEN_API __attribute__((visibility("default")))
#endif
#else
#ifdef _WIN32
#define BYTE_TRACK_EIGEN_API __declspec(dllimport)
#else
#define BYTE_TRACK_EIGEN_API
#endif
#endif