package com.sample.edgedetection.processor

import org.opencv.core.Point
import org.opencv.core.Size

data class Corners(var corners: List<Point>, var size: Size)