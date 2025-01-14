# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

# sed -En 's/^([^#][^_].+) = .*$/"\1"/p' constants.py | sort -u | paste -sd, - | sed 's/^/__all__ = [/; s/$/]/'
__all__ = ["_1080p","_1080p_tall","_2p5k","_4k","_4k_tall","_5k","_640wide","_720p","RC_CAR_CLASSES"]

_5k = (5312, 2988)
_4k = (3840, 2160)
_2p5k = (2560, 1440)
_1080p = (1920, 1080)
_720p = (1280, 720)

_4k_tall = (2160, 3840)
_1080p_tall = (1080, 1920)

_640wide = (640, 360)

RC_CAR_CLASSES = ['RC Car', '1/8 Scale', 'Stadium Truck', 'Short Course Truck']
