package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import java.util.Collections
import java.util.Comparator
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.core.CvType


const val TAG: String = "PaperProcessor"
const val MAGIC_RATE = 500

fun processPicture(previewFrame: Mat, isDetectEdge: Boolean): Corners? {
    var contours = findContours(previewFrame)
    var corners = getCorners(contours, previewFrame.size())

//    if (corners == null) {
//        contours = findContoursNoGray(previewFrame)
//        corners = getCorners(contours, previewFrame.size())
//    }

    if (corners != null && corners.corners != null) {
        var rescaledPoints: ArrayList<Point> = arrayListOf()
        var ratio = previewFrame.size().height / MAGIC_RATE
        if (!isDetectEdge)
            ratio = 1.0

        for (index in 0..corners.corners.size - 1) {
            val x = corners.corners[index].x * ratio
            val y = corners.corners[index].y * ratio

            rescaledPoints.add(Point(x, y))
        }
        corners.corners = rescaledPoints
        return corners
    }
    return corners
}

fun cropPicture(picture: Mat, pts: List<Point>): Mat {

    pts.forEach { Log.i(TAG, "point: $it") }
    val tl = pts[0]
    val tr = pts[1]
    val br = pts[2]
    val bl = pts[3]

    val widthA = sqrt((br.x - bl.x).pow(2.0) + (br.y - bl.y).pow(2.0))
    val widthB = sqrt((tr.x - tl.x).pow(2.0) + (tr.y - tl.y).pow(2.0))

    val dw = max(widthA, widthB)
    val maxWidth = java.lang.Double.valueOf(dw).toInt()


    val heightA = sqrt((tr.x - br.x).pow(2.0) + (tr.y - br.y).pow(2.0))
    val heightB = sqrt((tl.x - bl.x).pow(2.0) + (tl.y - bl.y).pow(2.0))

    val dh = max(heightA, heightB)
    val maxHeight = java.lang.Double.valueOf(dh).toInt()

    val croppedPic = Mat(maxHeight, maxWidth, CvType.CV_8UC4)

    val srcMat = Mat(4, 1, CvType.CV_32FC2)
    val dstMat = Mat(4, 1, CvType.CV_32FC2)

    srcMat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    dstMat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh)

    val m = Imgproc.getPerspectiveTransform(srcMat, dstMat)

    Imgproc.warpPerspective(picture, croppedPic, m, croppedPic.size())
    m.release()
    srcMat.release()
    dstMat.release()
    Log.i(TAG, "crop finish")
    return croppedPic
}

fun enhancePicture(src: Bitmap?): Bitmap {
    val srcMat = Mat()
    Utils.bitmapToMat(src, srcMat)
    Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGBA2GRAY)
    Imgproc.adaptiveThreshold(
        srcMat,
        srcMat,
        255.0,
        Imgproc.ADAPTIVE_THRESH_MEAN_C,
        Imgproc.THRESH_BINARY,
        15,
        15.0
    )
    val result = Bitmap.createBitmap(src?.width ?: 1080, src?.height ?: 1920, Bitmap.Config.RGB_565)
    Utils.matToBitmap(srcMat, result, true)
    srcMat.release()
    return result
}

private fun findContours2(src: Mat): List<MatOfPoint> {

    val grayImage: Mat
    val cannedImage: Mat
    val kernel: Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
    val dilate: Mat
    val size = Size(src.size().width, src.size().height)
    grayImage = Mat(size, CvType.CV_8UC4)
    cannedImage = Mat(size, CvType.CV_8UC1)
    dilate = Mat(size, CvType.CV_8UC1)

    Imgproc.cvtColor(src, grayImage, Imgproc.COLOR_BGR2GRAY)
    Imgproc.GaussianBlur(grayImage, grayImage, Size(5.0, 5.0), 0.0)
    Imgproc.threshold(grayImage, grayImage, 20.0, 255.0, Imgproc.THRESH_TRIANGLE)
    Imgproc.Canny(grayImage, cannedImage, 75.0, 200.0)
    Imgproc.dilate(cannedImage, dilate, kernel)
    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(
        dilate,
        contours,
        hierarchy,
        Imgproc.RETR_TREE,
        Imgproc.CHAIN_APPROX_SIMPLE
    )

    val filteredContours = contours
        .filter { p: MatOfPoint -> Imgproc.contourArea(p) > 100e2 }
        .sortedByDescending { p: MatOfPoint -> Imgproc.contourArea(p) }
        .take(25)


    Log.i("FILTERED COUNT", filteredContours.size.toString())

    hierarchy.release()
    grayImage.release()
    cannedImage.release()
    kernel.release()
    dilate.release()

    return filteredContours
}

private fun findContoursNoGray(src: Mat): List<MatOfPoint> {
    val grayImage: Mat
    val cannedImage: Mat
    val resizedImage: Mat

    val ratio: Double = src.size().height / MAGIC_RATE
    val height = (src.size().height / ratio)
    val width = (src.size().width / ratio)
    val size = Size(width, height)

    resizedImage = Mat(size, CvType.CV_8UC4)
    cannedImage = Mat(size, CvType.CV_8UC1)

    Imgproc.resize(src, resizedImage, size)
    Imgproc.GaussianBlur(resizedImage, resizedImage, Size(5.0, 5.0), 0.0)
    Imgproc.Canny(resizedImage, cannedImage, 80.0, 100.0, 3, false)

    val contours: ArrayList<MatOfPoint> = ArrayList<MatOfPoint>()
    val hierarchy = Mat()

    Imgproc.findContours(cannedImage, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
    hierarchy.release()


    val filteredContours = contours
        .sortedByDescending { p: MatOfPoint -> Imgproc.contourArea(p) }

    resizedImage.release()
    cannedImage.release()

    return filteredContours
}

private fun findContours(src: Mat): List<MatOfPoint> {
    val grayImage: Mat
    val cannedImage: Mat
    val resizedImage: Mat

    val ratio: Double = src.size().height / MAGIC_RATE
    val height = (src.size().height / ratio)
    val width = (src.size().width / ratio)
    val size = Size(width, height)
//    val size = Size(src.size().width, src.size().height)

    resizedImage = Mat(size, CvType.CV_8UC4)
    grayImage = Mat(size, CvType.CV_8UC4)
    cannedImage = Mat(size, CvType.CV_8UC1)

    Imgproc.resize(src, resizedImage, size)

    // Gray
//    Imgproc.cvtColor(resizedImage, grayImage, Imgproc.COLOR_RGBA2GRAY, 4)
//    Imgproc.GaussianBlur(grayImage, grayImage, Size(5.0, 5.0), 0.0)
//    Imgproc.Canny(grayImage, cannedImage, 80.0, 100.0, 3, false)
//

    val gray = Mat(resizedImage.rows(), resizedImage.cols(), src.type())

    Imgproc.cvtColor(resizedImage, gray, Imgproc.COLOR_BGR2GRAY)
    Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

    val kernel = Imgproc.getStructuringElement(
        Imgproc.MORPH_ELLIPSE, Size(
            5.0,
            5.0
        )
    )
    Imgproc.morphologyEx(
        gray,
        gray,
        Imgproc.MORPH_CLOSE,
        kernel
    ) // fill holes
    Imgproc.morphologyEx(
        gray,
        gray,
        Imgproc.MORPH_OPEN,
        kernel
    ) //remove noise
    Imgproc.dilate(gray, gray, kernel)

    val edges = Mat(src.rows(), src.cols(), src.type())
    Imgproc.Canny(gray, edges, 80.0, 100.0, 3, false)

    val contours: ArrayList<MatOfPoint> = ArrayList<MatOfPoint>()
    val hierarchy = Mat()

    Imgproc.findContours(
        edges, contours, hierarchy, Imgproc.RETR_TREE,
        Imgproc.CHAIN_APPROX_SIMPLE
    )

//    Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
    hierarchy.release()

    val filteredContours = contours
            .sortedByDescending { p: MatOfPoint -> Imgproc.contourArea(p) }

    resizedImage.release()
    grayImage.release()
    cannedImage.release()

    return filteredContours
}

private fun getCorners(contours: List<MatOfPoint>, srcSize: Size): Corners? {
    val ratio = srcSize.height / MAGIC_RATE
    val height = (srcSize.height / ratio)
    val width = (srcSize.width / ratio)
    val size = Size(width, height)

    for (index in 0..contours.size - 1) {
            val c2f = MatOfPoint2f(*contours[index].toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true)

            val points = approx.toArray().asList()
            val foundPoints = sortPoints(points)

            // select biggest 4 angles polygon
            if (insideArea(foundPoints, size)) {


                return Corners(foundPoints, size)
            }
    }

    return null
}

private fun checkDistances(points: List<Point>): Boolean {
    val distanceThreshold = 200.0
    var hasOkDistance = true
    for (i in 0..points.size - 1) {
        for (j in i + 1..points.size - 1) {
            val distance = getDistance(points[i], points[j])
            if (distance < distanceThreshold) {
                hasOkDistance = false
                break
            }
        }
    }
    return hasOkDistance
}

fun getDistance(p1: Point, p2: Point): Double {
    return Math.sqrt(
        Math.pow(p2.x - p1.x, 2.0)
            +
            Math.pow(p2.y - p1.y, 2.0)
    )
}

private fun sortPoints(points: List<Point>): List<Point> {
    val p0 = points.minByOrNull { point -> point.x + point.y } ?: Point()
    val p1 = points.minByOrNull { point: Point -> point.y - point.x } ?: Point()
    val p2 = points.maxByOrNull { point: Point -> point.x + point.y } ?: Point()
    val p3 = points.maxByOrNull { point: Point -> point.y - point.x } ?: Point()
    return listOf(p0, p1, p2, p3)
}

private fun insideArea(rp: List<Point>, size: Size): Boolean {
    val height = size.height
    val width = size.width

    val minimumSize = width / 10

    val isANormalShape = rp[0].x !== rp[1].x && rp[1].y !== rp[0].y && rp[2].y !== rp[3].y && rp[3].x !== rp[2].x
    val isBigEnough = (rp[1].x - rp[0].x >= minimumSize && rp[2].x - rp[3].x >= minimumSize
            && rp[3].y - rp[0].y >= minimumSize && rp[2].y - rp[1].y >= minimumSize)

    val leftOffset = rp[0].x - rp[3].x
    val rightOffset = rp[1].x - rp[2].x
    val bottomOffset = rp[0].y - rp[1].y
    val topOffset = rp[2].y - rp[3].y

    val isAnActualRectangle = (leftOffset <= minimumSize && leftOffset >= -minimumSize
            && rightOffset <= minimumSize && rightOffset >= -minimumSize
            && bottomOffset <= minimumSize && bottomOffset >= -minimumSize
            && topOffset <= minimumSize && topOffset >= -minimumSize)

    return isANormalShape && isAnActualRectangle && isBigEnough
}

