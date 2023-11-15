#ifdef BUILDING_BYTE_TRACK_EIGEN
#define BYTE_TRACK_EIGEN_API __declspec(dllexport)
#else
#define BYTE_TRACK_EIGEN_API __declspec(dllimport)
#endif