#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <Python.h>

using namespace std;
using namespace cv;

Point fishEye1,fishEye2,fishCenter, fishHead;
Mat cur_img,org_img;
bool draw;
Mat roi;
Point tailPt_a, tailPt_b, topEnd;
Point cursor;
int threshold_val = 22;
Rect rect;
/*
class UserInterface
{
public:
	static void on_trackbar_setThreshold(int, void*) {
		int max_val = 255;
		Mat binaryzation = Mat::zeros(cur_img.size(), CV_8UC1);
		threshold(cur_img, binaryzation, threshold_val, max_val, CV_THRESH_BINARY);

		imshow("setThreshold", binaryzation);

	}

	bool setThreshold(Mat org) {
		namedWindow("setThreshold", CV_WINDOW_NORMAL);
		createTrackbar("Threshold", "setThreshold", &threshold_val, 255, on_trackbar_setThreshold);
		on_trackbar_setThreshold(threshold_val, 0);
		while (char(waitKey(1)) != 'q') {}
		return true;
	}

	void on_mouse_findHeadAndCenter(int event, int x, int y, int flags, void* ustc) {
		if (event == CV_EVENT_LBUTTONDBLCLK) {
			fishEye1 = Point(x, y);
			circle(cur_img, fishEye1, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
			imshow("findHeadAndCenter", cur_img);
		}
		else if (event == CV_EVENT_RBUTTONDBLCLK) {

			fishEye2 = Point(x, y);
			fishHead = (fishEye1 + fishEye2) / 2;
			circle(cur_img, fishHead, 2, Scalar(0, 0, 0, 255), CV_FILLED, CV_AA, 0);
			circle(cur_img, fishEye2, 2, Scalar(0, 255, 0, 0), CV_FILLED, CV_AA, 0);
			imshow("findHeadAndCenter", cur_img);

		}
		else if (event == CV_EVENT_MBUTTONDOWN) {

			fishCenter = Point(x, y);
			circle(cur_img, fishCenter, 2, Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
			imshow("findHeadAndCenter", cur_img);
			cout << "fishCenter:" << fishCenter.x << ',' << fishCenter.y << endl;
			cout << "fishHead:" << fishHead.x << ',' << fishHead.y << endl;
		}

	}

	bool findHeadAndCenter(Mat org) {
		namedWindow("findHeadAndCenter", CV_WINDOW_NORMAL);
		imshow("findHeadAndCenter", org);
		setMouseCallback("findHeadAndCenter", on_mouse_findHeadAndCenter, 0);
		while (char(waitKey(1)) != 'q') {}
		return true;
	}

};
*/
static void on_trackbar_setThreshold(int, void*) {
	int max_val = 255;
	Mat binaryzation = Mat::zeros(cur_img.size(), CV_8UC1);
	threshold(cur_img, binaryzation, threshold_val, max_val, CV_THRESH_BINARY);

	imshow("setThreshold", binaryzation);

}

bool setThreshold() {
	namedWindow("setThreshold", CV_WINDOW_NORMAL);
	createTrackbar("Threshold", "setThreshold", &threshold_val, 255, on_trackbar_setThreshold);
	on_trackbar_setThreshold(threshold_val, 0);
	cout << "Press 'q' to exit." << endl;
	while (char(waitKey(1)) != 'q') {}
	return true;
}

void on_mouse_findHeadAndCenter(int event, int x, int y, int flags, void* ustc) {
	if (event == CV_EVENT_LBUTTONDBLCLK) {
		fishEye1 = Point(x, y);
		circle(cur_img, fishEye1, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		imshow("findHeadAndCenter", cur_img);
	}
	else if (event == CV_EVENT_RBUTTONDBLCLK) {

		fishEye2 = Point(x, y);
		fishHead = (fishEye1 + fishEye2) / 2;
		circle(cur_img, fishHead, 2, Scalar(0, 0, 0, 255), CV_FILLED, CV_AA, 0);
		circle(cur_img, fishEye2, 2, Scalar(0, 255, 0, 0), CV_FILLED, CV_AA, 0);
		imshow("findHeadAndCenter", cur_img);

	}
	else if (event == CV_EVENT_MBUTTONDOWN) {

		fishCenter = Point(x, y);
		circle(cur_img, fishCenter, 2, Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		imshow("findHeadAndCenter", cur_img);
		cout << "fishCenter:" << fishCenter.x << ',' << fishCenter.y << endl;
		cout << "fishHead:" << fishHead.x << ',' << fishHead.y << endl;
	}

}

bool findHeadAndCenter() {
	namedWindow("findHeadAndCenter", CV_WINDOW_NORMAL);
	imshow("findHeadAndCenter", cur_img);
	setMouseCallback("findHeadAndCenter", on_mouse_findHeadAndCenter, 0);
	cout << "Double-click the left mouse button to select one eye." << endl;
	cout << "Double-click the right mouse button to select another eye." << endl;
	cout << "Click the middle mouse button to select center." << endl;
	cout << "Press 'q' to exit." << endl;
	while (char(waitKey(1)) != 'q') {}
	return true;
}

/*Find the closest point on the contour to the reference point, return the index findClosestPt*/
int findClosestPt(vector<Point>& contour, Point point)
{
	double minDist = 1000000;
	double tempDist = 0;
	Point temp;
	int goodIndex = 0;
	for (int i = 0; i < contour.size(); i++)
	{
		temp = contour[i];
		tempDist = norm(contour[i] - point);

		if (tempDist < minDist)
		{
			goodIndex = i;
			minDist = tempDist;
		}
	}

	return goodIndex;
}


/* Get the Euclidean distance from point P to line AB */
double getPt2LineDistance(Point2f P, Point2f A, Point2f B)
{
	Point2f BA = B - A; // the vector from A to B
	//Point2f PA = P - A; // the vector from A to P
	double dist = abs(BA.y * P.x - BA.x * P.y + B.cross(A)) / norm(BA);
	return dist;
}

/* Find 2 intersection points of a line (AB) and contour */
vector<int> findPtsLineIntersectContour(vector<Point> & contour, Point2f A, Point2f B)
{
	vector<int> goodIndices(2);
	Point2f curPt; // current point
	vector<Point> ptList; // store potential intersection points
	vector<int> idxList; // store indices of potential intersection points
	double distThre = 1.0; // threshold to decide whether it is an intersection pt
	for (int i = 0; i < contour.size(); i++)
	{
		curPt = contour[i];
		double pt2line = getPt2LineDistance(curPt, A, B);
		if (pt2line < distThre)
		{
			ptList.push_back(contour[i]);
			idxList.push_back(i);
		}

	}

	// assign intersection points to each side
	int idxA = findClosestPt(ptList, A); // the one closest to A
	int idxB = findClosestPt(ptList, B); // the one closest to B

	if (idxA < idxB)
	{
		goodIndices[0] = idxList[idxA];
		goodIndices[1] = idxList[idxB];
	}
	else
	{
		goodIndices[0] = idxList[idxB];
		goodIndices[1] = idxList[idxA];
	}


	return goodIndices;

}


/*This function return a radian to describe the fishtailing motion */
bool fishAngleAnalysis(Mat fishImg, Point fishHead, Point fishCenter, Point * fishTail_return, double* fishAngle,int threshold_val) {
	//Find the contour of fish
	Mat binaryzation;
	double  max_val = 255, maxFishArea = 15000, minFishArea = 2000;
	vector<vector<Point>> allContours, fishContours;
	threshold(fishImg, binaryzation, threshold_val, max_val, CV_THRESH_BINARY);
	findContours(binaryzation, allContours, CV_RETR_LIST, CHAIN_APPROX_NONE);
	for (int i = 0; i < allContours.size(); i++) {
		if (contourArea(allContours[i]) < maxFishArea && contourArea(allContours[i]) > minFishArea)
			fishContours.push_back(allContours[i]);
	}
	if (fishContours.size() != 1) {
		cout << "Can't find contour of fish!Area of all contours:";
		for (int i = 0; i < allContours.size(); i++) {
			cout << contourArea(allContours[i]) << ',';
		}
		cout << endl;
		return false;
	}

	//Find the tail of fish

	double Pt2center = norm(fishContours[0][0] - fishCenter);
	topEnd = fishContours[0][0];

	for (int i = 1; i < fishContours[0].size(); i++)
	{
		double curPt2center = norm(fishContours[0][i] - fishCenter);
		if (Pt2center < curPt2center) {
			topEnd = fishContours[0][i];
			Pt2center = curPt2center;
			circle(fishImg, topEnd, 1, Scalar(255), -1);

		}


	}
	Point tailAxis = topEnd - fishCenter;
	tailPt_a = fishCenter + tailAxis * 9 / 10 + Point(tailAxis.y, -tailAxis.x)/4;
	tailPt_b = fishCenter + tailAxis * 9 / 10 + Point(-tailAxis.y, tailAxis.x)/4;
	vector<int> fishTail = findPtsLineIntersectContour(fishContours[0], tailPt_a, tailPt_b);

	//Calculate the angle
	Point fishHeadVector, fishTailVector;
	fishHeadVector = fishCenter - fishHead;
	fishTailVector = (fishContours[0][fishTail[0]]+ fishContours[0][fishTail[1]])/2 - fishCenter;
	double sinfi;
	sinfi = -(fishHeadVector.x * fishTailVector.y - fishTailVector.x * fishHeadVector.y) / (norm(fishHeadVector) * norm(fishTailVector));
	*fishAngle = asin(sinfi);
	*fishTail_return = (fishContours[0][fishTail[0]] + fishContours[0][fishTail[1]]) / 2;
	//drawContours(fishImg, fishContours, -1, Scalar(255),2);
	//imshow("contour", fishImg);
	return true;
}


int predict_left(double* boutStart) {
	Py_Initialize();
	if (Py_IsInitialized() == 0) {
		cout << "Py_Initialize failed." << endl;
	}
	PyObject* pModule = PyImport_ImportModule("predict");
	if (pModule == NULL)
		cout << "Py_ImportModule failed." << endl;
	PyObject * pFunc = PyObject_GetAttrString(pModule, "predict_left");
	PyObject * PyList = PyList_New(40);
	PyObject * ArgList = PyTuple_New(1);
	for (int Index_i = 0; Index_i < PyList_Size(PyList); Index_i++) {
		PyList_SetItem(PyList, Index_i, PyFloat_FromDouble(boutStart[Index_i]));
	}
	PyTuple_SetItem(ArgList, 0, PyList);
	PyObject* pReturn = NULL;
	pReturn = PyObject_CallObject(pFunc, ArgList);
	int result;
	PyArg_Parse(pReturn, "i", &result);
	cout << "predict:" << result << endl;
	Py_Finalize();
	return result;

}

bool fishAngleAnalysis_test(String fishVideoAddress, bool isGrey) {
	VideoCapture capture(fishVideoAddress);
	Point testTail;
	Mat curImg;
	VideoWriter writer;
	writer.open("F:\\fishData\\new_stream5.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, Size(400, 400), true);
	namedWindow("output", CV_WINDOW_NORMAL);
	//namedWindow("org", CV_WINDOW_NORMAL);
	capture >> curImg;
	//findFishHeadAndCenter(curImg);
	//getchar();
	for (int i = 0; i < 780; i++) {
		capture >> curImg;
	}


	int n = 0;
	int checkPoint=0;
	int boutStart = 0;
	int predict;
	for (int i = 0; i < 100; i++) {

		Mat grey;
		Point fishTail = Point(-1, -1);
		double fishAngle[10000];
		capture >> curImg;

		if (!isGrey) {
			cvtColor(curImg, grey, CV_BGR2GRAY);
			if (!fishAngleAnalysis(grey, fishHead, fishCenter, &fishTail, &fishAngle[i], threshold_val)) {
				cout << "AngleAnalysis error!" << endl;
				return false;
			}
		}
		else {
			if (!fishAngleAnalysis(curImg, fishHead, fishCenter, &fishTail, &fishAngle[i], threshold_val)) {
				cout << "AngleAnalysis error!" << endl;
				return false;
			}
		}
		/*

		if (n == checkPoint) {
			if (boutStart > 0) {
				predict=predict_left(&fishAngle[boutStart]);
				cout << "predict:" << predict << endl;
				boutStart = 0;
			}

			if (fishAngle[i] > 0.2|| fishAngle[i] < -0.2) {
				boutStart = i - 4;
				checkPoint = checkPoint + 40;
				cout << "Find bout!" << endl;
			}
		}
		else
			n++;
		*/
		cout << "fishAngle:" << fishAngle[i] << endl;
		circle(curImg, fishHead, 1, Scalar(0, 0, 0, 255), -1);
		circle(curImg, fishCenter, 1, Scalar(0, 0, 255, 0), -1);
		circle(curImg, fishTail, 1, Scalar(255, 0, 0), -1);
		circle(curImg, tailPt_a, 1, Scalar(255, 0, 0), -1);
		circle(curImg, tailPt_b, 1, Scalar(255, 0, 0), -1);
		circle(curImg, topEnd, 1, Scalar(255, 0, 0), -1);

		line(curImg, fishHead, fishCenter, Scalar(0, 255, 0));
		line(curImg, fishTail, fishCenter, Scalar(0, 255, 0));
		imshow("output", curImg);
		writer << curImg;
		//waitKey();
	}
	writer.release();
	return true;
}

bool videoEditing(String fishVideoAddress, int start_frame) {
	VideoCapture cap(fishVideoAddress);
	float fps = cap.get(CV_CAP_PROP_FPS);
	long framewidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	long framehigh = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	float framecount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	VideoWriter writer;
	writer.open("F:/fishData/new_stream1.mpg", CV_FOURCC('D', 'I', 'V', 'X'), 10, Size(248, 250), false);
	int i = 0;
	while (1) {
		Mat frame;
		cap >> frame;
		i++;
		if (frame.empty())break;
		if (i >= start_frame) {
			writer << frame;
		}
	}
	return true;

}

int main() {
	//
	VideoCapture capture("F://fishData//test5.avi");
	Point testTail;
	Mat curImg;

	capture >> curImg;

	curImg.copyTo(cur_img);
	findHeadAndCenter();
	setThreshold();

	fishAngleAnalysis_test("F://fishData//test5.avi",  false);

	waitKey();

	//videoEditing("F://fishData//test3.avi", 1400);
}
