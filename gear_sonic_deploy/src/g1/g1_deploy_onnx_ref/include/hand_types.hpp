/**
 * @file hand_types.hpp
 * @brief Shared 6-DOF Inspire hand types and URDF-limit helpers.
 */

#ifndef HAND_TYPES_HPP
#define HAND_TYPES_HPP

#include <algorithm>
#include <array>
#include <cstddef>

namespace hand {

static constexpr std::size_t HAND_DOF = 6;

using HandJointArray = std::array<double, HAND_DOF>;

// Internal policy/URDF order:
// [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
static constexpr HandJointArray INSPIRE_URDF_LOWER = {
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0
};

static constexpr HandJointArray INSPIRE_URDF_UPPER = {
    1.3, 0.6, 1.7, 1.7, 1.7, 1.7
};

static inline HandJointArray inspireOpenPoseRad() {
    return INSPIRE_URDF_LOWER;
}

static inline HandJointArray inspireClosedPoseRad() {
    return INSPIRE_URDF_UPPER;
}

static inline double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

static inline double radToInspireCommand(double value, std::size_t idx) {
    const double lower = INSPIRE_URDF_LOWER[idx];
    const double upper = INSPIRE_URDF_UPPER[idx];
    const double x = clamp01((value - lower) / (upper - lower));
    return 1.0 - x;
}

static inline double inspireCommandToRad(double command, std::size_t idx) {
    const double lower = INSPIRE_URDF_LOWER[idx];
    const double upper = INSPIRE_URDF_UPPER[idx];
    return lower + (1.0 - clamp01(command)) * (upper - lower);
}

}  // namespace hand

#endif  // HAND_TYPES_HPP
