#ifndef LOADTERMINAL_HPP
#define LOADTERMINAL_HPP

#include <QtCore>
#include <iostream>
#include <sys/stat.h>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <zbar.h>


using namespace std;
using namespace cv;
using namespace dlib;
using namespace zbar;
using namespace aruco;

class LoadTerminal : public QObject{
    Q_OBJECT
private:
    VideoCapture cap; // Enter camera device number can be 0,1 etc.
    cv::CascadeClassifier face_cascade;
    dlib::frontal_face_detector detector;
    dlib::shape_predictor sp;
    ImageScanner scanner;
    std::vector<int> ids;
    std::vector< std::vector< Point2f> > corners;
    cv::Ptr<aruco::Dictionary> dictionary;
    string cmd;

    // Defining the commands
    enum commands {
        help,
        vPlay,vSave,vPreview,
        nPeople,faceDetect,faceCompare,faceMerge,faceLandmarks,imageComic,
        barScan,QRScan,arucoScan,
        unknown
        // Add your own commands here..
    };

    // Converts commands into their hash value
    commands hash_cmd(string const& inString){
        if (inString == "help") return help;
        if (inString == "vPlay") return vPlay;
        if (inString == "vSave") return vSave;
        if (inString == "vPreview") return vPreview;
        if (inString == "nPeople") return nPeople;
        if (inString == "faceDetect") return faceDetect;
        if (inString == "faceCompare") return faceCompare;
        if (inString == "faceMerge") return faceMerge;
        if (inString == "faceLandmarks") return faceLandmarks;
        if (inString == "imageComic") return imageComic;
        if (inString == "barScan") return barScan;
        if (inString == "QRScan") return QRScan;
        if (inString == "arucoScan") return arucoScan;

        return unknown;
    }

    // Check if system path exists
    inline bool exists_test(const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
    }

    // Read points stored in the text files
    std::vector<Point2f> readPoints(string pointsFileName){
        std::vector<Point2f> points;
        ifstream ifs(pointsFileName);
        float x, y;
        while(ifs >> x >> y)
        {
            points.push_back(cv::Point2f(x,y));
        }

        return points;
    }
    // Apply affine transform calculated using srcTri and dstTri to src
    void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
    {

        // Given a pair of triangles, find the affine transform.
        cv::Mat warpMat = getAffineTransform( srcTri, dstTri );

        // Apply the Affine Transform just found to the src image
        warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
    }

    // Warps and alpha blends triangular regions from img1 and img2 to img
    void morphTriangle(Mat &img1, Mat &img2, Mat &img, std::vector<Point2f> &t1, std::vector<Point2f> &t2, std::vector<Point2f> &t, double alpha)
    {

        // Find bounding rectangle for each triangle
        cv::Rect r = boundingRect(t);
        cv::Rect r1 = boundingRect(t1);
        cv::Rect r2 = boundingRect(t2);

        // Offset points by left top corner of the respective rectangles
        std::vector<Point2f> t1Rect, t2Rect, tRect;
        std::vector<Point> tRectInt;
        for(int i = 0; i < 3; i++)
        {
            tRect.push_back( Point2f( t[i].x - r.x, t[i].y -  r.y) );
            tRectInt.push_back( Point(t[i].x - r.x, t[i].y - r.y) ); // for fillConvexPoly

            t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
            t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        }

        // Get mask by filling triangle
        Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
        fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

        // Apply warpImage to small rectangular patches
        Mat img1Rect, img2Rect;
        img1(r1).copyTo(img1Rect);
        img2(r2).copyTo(img2Rect);

        Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
        Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

        applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
        applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

        // Alpha blend rectangular patches
        Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

        // Copy triangular region of the rectangular patch to the output image
        multiply(imgRect,mask, imgRect);
        multiply(img(r), Scalar(1.0,1.0,1.0) - mask, img(r));
        img(r) = img(r) + imgRect;


    }

public:
    LoadTerminal(QObject *parent = 0) : QObject(parent) {}
    LoadTerminal(){
        cap.open(0);
        detector = get_frontal_face_detector();
        deserialize("Others/shape_predictor_68_face_landmarks.dat") >> sp;
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    }
public slots:
    // Handling the command line
    void run(){
        // Initializing
        cap.open(0);
        detector = get_frontal_face_detector();
        deserialize("Others/shape_predictor_68_face_landmarks.dat") >> sp;
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

        cout<<"Welcome to IPTerminal .."<<endl;
        cout<<"Type 'help' to know about commands.."<<endl;
        while(1){
            cout<<">>";
            cin>>cmd;
            switch(hash_cmd(cmd)){
            case help:
            {
                cout<<"***********************************************************************"<<endl;
                cout<<"BASIC COMMANDS:\n"<<endl;
                cout<<"* LIVE FOOTAGE ANALYSIS : vPlay\n"<<endl;
                cout<<"Saving Video Feed : vSave"<<endl;
                cout<<"Previewing Videos : vPreview\n"<<endl;
                cout<<"* HUMAN DETECTION : \n"<<endl;
                cout<<"Number Of People : nPeople"<<endl;
                cout<<"Face Detection : faceDetect\n"<<endl;
                cout<<"* FACIAL ANALYSIS :\n"<<endl;
                cout<<"Compare 2 faces : faceCompare"<<endl;
                cout<<"Merge 2 faces : faceMerge"<<endl;
                cout<<"View facial landmarks : faceLandmarks\n"<<endl;
                cout<<"* IMAGE MANIPULATION :\n"<<endl;
                cout<<"Convert Image into Comic : imageComic\n"<<endl;
                cout<<"* MARKER DETECTION : \n"<<endl;
                cout<<"Bar Code Scanning : barScan"<<endl;
                cout<<"Aruco Scanning : arucoScan"<<endl;
                cout<<"QR Scanning : QRScan\n"<<endl;
                cout<<"************************************************************************"<<endl;
                break;
            }

            case vPlay:
            {
                cout<<"Live Footage starting (Press Esc to stop)"<<endl;
                // Insert your code here..
                Mat frame;
                while(1){
                    // Capture frame-by-frame
                    cap >> frame;

                    // If the frame is empty, break immediately
                    if (frame.empty())
                        break;
                    imshow("Live Footage", frame);
                    char c=(char)waitKey(25);
                    if(c==27)
                        break;
                }
                cap.release();
                break;
            }
            case vSave:
            {
                cout<<"Saving Current Video Feed (Press Esc to stop)"<<endl;
                // Insert your code here..
                if(!cap.isOpened())
                {
                    cout << "Error opening video stream" << endl;
                    break;
                }
                // Default resolution of the frame is obtained.The default resolution is system dependent.
                int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
                int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
                // Initialize the video writer
                VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height));
                while(1)
                {
                    Mat frame;

                    // Capture frame-by-frame
                    cap >> frame;

                    // If the frame is empty, break immediately
                    if (frame.empty())
                        break;

                    // Write the frame into the file 'outcpp.avi'
                    video.write(frame);

                    // Display the resulting frame
                    imshow( "Frame", frame );

                    // Press  ESC on keyboard to  exit
                    char c = (char)waitKey(1);
                    if( c == 27 )
                        break;
                }

                // When everything done, release the video capture and write object
                cap.release();
                video.release();

                break;
            }
            case vPreview:
            {
                string path;
                cout<<"Select Video to Preview :"<<endl;
                // Insert your code here..
                cout<<"Enter Path : ";
                cin>>path;
                cap.open(path);
                if(!cap.isOpened())
                {
                    cout << "Error opening video stream" << endl;
                    break;
                }
                cout<<"Opening file (Press Esc to stop the video)"<<endl;
                Mat frame;
                while(1){
                    cap >> frame;
                    if (frame.empty())
                        break;
                    imshow("Preview Footage", frame);
                    char c=(char)waitKey(25);
                    if(c==27)
                        break;
                }
                cap.release();
                break;
            }
            case nPeople:
            {
                cout<<"Human Detection Activated (Press Esc to stop)"<<endl;
                // Insert your code here..
                Mat frame;
                HOGDescriptor hog;
                hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
                int count = 0;
                while(1){
                    cap>>frame;
                    if(frame.empty()){
                        break;
                    }
                    Mat img = frame.clone();
                    std::vector<Rect> found;
                    std::vector<double> weights;
                    // Detect people from frames
                    hog.detectMultiScale(img, found, weights,0,Size(4,4));

                    /// draw detections and store location
                    for( size_t i = 0; i < found.size(); i++ )
                    {
                        if(weights[i]>=1){
                            count++;
                            cv::rectangle(img, found[i], cv::Scalar(0,0,255), 3);
                            stringstream temp;
                            temp << weights[i];
                            putText(img, temp.str(),Point(found[i].x,found[i].y+50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
                        }
                    }
                    imshow("Human Detection",img);
                    char c=(char)waitKey(25);
                    if(c==27)
                        break;
                }
                cap.release();
                cout<<"Number of People detected :"<<count<<endl;
                break;
            }
            case faceDetect:
            {
                cout<<"Face Detection Activated.."<<endl;
                // Insert your code here..
                // Load Haar Classifiers file
                if( !face_cascade.load( "Others/haarcascade_frontalface_default.xml") ) printf("--(!)Error loading\n");
                Mat frame,frame_gray;
                std::vector<Rect> faces;
                while(1){
                    cap>>frame;
                    cvtColor(frame,frame_gray,COLOR_BGR2GRAY);
                    equalizeHist( frame_gray, frame_gray );
                    // Detect faces
                    face_cascade.detectMultiScale( frame_gray, faces, 1.4, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
                    for(size_t i=0;i<faces.size();i++){
                        // Draw detected faces
                        cv::rectangle(frame,faces[i],Scalar(0,255,0),2,8);
                    }
                    imshow("Face Detection",frame);
                    char c=(char)waitKey(25);
                    if(c==27)
                        break;
                }
                cap.release();
                break;
            }
            case faceCompare:
            {
                cout<<"Face Compare Activated.."<<endl;
                // Insert your code here..
                string path1,path2;
                cout<<"Enter image paths :";
                cin>>path1>>path2;
                if(exists_test(path1) && exists_test(path2)){
                    Mat img1 = imread(path1,0);
                    Mat img2 = imread(path2,0);
                    // Load Haar classifier file
                    if( !face_cascade.load( "Others/haarcascade_frontalface_default.xml") ) printf("--(!)Error loading\n");
                    equalizeHist(img1,img1);
                    equalizeHist(img2,img2);
                    // Detecting faces
                    std::vector<Rect> faces1,faces2;
                    face_cascade.detectMultiScale( img1, faces1, 1.4, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
                    face_cascade.detectMultiScale( img2, faces2, 1.4, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
                    // Comparing faces
                    if(faces1.size() && faces2.size()){
                        Mat face1 = img1(faces1[0]);
                        Mat face2 = img2(faces2[0]);
                        threshold( face1, face1, 100,255,THRESH_BINARY );
                        threshold( face2, face2, 100,255,THRESH_BINARY );
                        resize(face1,face1,Size(100,100));
                        resize(face2,face2,Size(100,100));
                        Mat result;
                        compare(face1,face2,result,cv::CMP_EQ);
                        float totalPixels = result.cols * result.rows;
                        float percentage  = (float)(countNonZero(result)/totalPixels)*100;
                        cout<<"Matching percentage is "<<percentage<<endl;
                    }
                    else{
                        cout<<"Face not detected."<<endl;
                    }

                }
                else
                    cout<<"Path entered does not exists!"<<endl;
                break;
            }
            case faceMerge:
            {
                cout<<"Face Merge Activated.."<<endl;
                // Insert your code here..
                string path1,path2;
                cout<<"Enter image paths :";
                cin>>path1>>path2;
                //alpha controls the degree of morph
                double alpha = 0.5;

                if(exists_test(path1) && exists_test(path2)){
                    Mat img1 = imread(path1);
                    Mat img2 = imread(path2);
                    //convert Mat to float data type
                    img1.convertTo(img1, CV_32F);
                    img2.convertTo(img2, CV_32F);


                    //empty average image
                    Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);


                    //Read landmark points from file(required beforehand for image)
                    std::vector<Point2f> points1 = readPoints( path1 + ".txt");
                    std::vector<Point2f> points2 = readPoints( path2 + ".txt");
                    std::vector<Point2f> points;

                    //compute weighted average point coordinates
                    for(size_t i = 0; i < points1.size(); i++)
                    {
                        float x, y;
                        x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
                        y =  ( 1 - alpha ) * points1[i].y + alpha * points2[i].y;

                        points.push_back(Point2f(x,y));

                    }


                    //Read triangle indices
                    ifstream ifs("tri.txt");
                    int x,y,z;

                    while(ifs >> x >> y >> z)
                    {
                        // Triangles
                        std::vector<Point2f> t1, t2, t;

                        // Triangle corners for image 1.
                        t1.push_back( points1[x] );
                        t1.push_back( points1[y] );
                        t1.push_back( points1[z] );

                        // Triangle corners for image 2.
                        t2.push_back( points2[x] );
                        t2.push_back( points2[y] );
                        t2.push_back( points2[z] );

                        // Triangle corners for morphed image.
                        t.push_back( points[x] );
                        t.push_back( points[y] );
                        t.push_back( points[z] );

                        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);

                    }

                    // Display Result
                    imshow("Morphed Face", imgMorph/255.0);
                    waitKey(0);

                }
                else
                    cout<<"Path entered does not exists!"<<endl;
                break;
            }
            case faceLandmarks:
            {
                cout<<"Face Landmarks Activated.."<<endl;
                // Insert your code here..
                string path;
                cout<<"Enter Image Path :";
                cin>>path;
                Mat temp = imread(path);
                cv_image<bgr_pixel> img(temp);
                //			dlib::load_image(img, path);

                // Now tell the face detector to give us a list of bounding boxes
                // around all the faces in the image.
                std::vector<dlib::rectangle> dets = detector(img);
                cout << "Number of faces detected: " << dets.size() << endl;
                std::vector<dlib::full_object_detection> shapes;
                for (unsigned long j = 0; j < dets.size(); ++j)
                {
                    dlib::full_object_detection shape = sp(img, dets[j]);
                    cout << "number of parts: "<< shape.num_parts() << endl;
                    shapes.push_back(sp(img,dets[j]));
                }

                // View our face poses on the screen.
                image_window win;
                win.set_image(img);
                win.add_overlay(dlib::render_face_detections(shapes));
                break;
            }
            case imageComic:
            {
                cout<<"Image Comic Activated.."<<endl;
                // Insert your code here..
                string path;
                cout<<"Enter Image path :";
                cin>>path;
                if(exists_test(path)){
                    Mat src = imread(path);
                    Mat gray,edges;
                    cvtColor(src,gray,CV_BGR2GRAY);
                    blur(gray,edges,Size(3,3),Point(-1,-1));
                    Canny(edges,edges,50,150,3,false);

                    // Making edges a bit fatter
                    Mat kernel = Mat::ones( 3, 3, CV_32F )/ 12.0;

                    /// Apply filter
                    filter2D(edges, edges, -1 , kernel, Point(-1,-1), 0, BORDER_DEFAULT );
                    threshold(edges,edges,50,255,0);

                    // Back to Color
                    cvtColor(edges,edges,CV_GRAY2BGR);

                    // Perform mean shift filtering
                    Mat shifted;
                    pyrMeanShiftFiltering(src,shifted,5,20);

                    // Substract both images for final effects
                    Mat res;
                    subtract(shifted,edges,res);

                    imshow("Comic Image",res);

                    waitKey(0);
                }
                else
                    cout<<"Path entered does not exists!"<<endl;
                break;
            }
            case barScan:
            {
                cout<<"Bar Code Scanning Activated.."<<endl;
                // Insert your code here..
                // You can use zbar library for barcode(QRcode) scanning

                string path;
                cout<<"Enter Image path :";
                cin>>path;
                if(exists_test(path)){
                    Mat src = imread(path,0);
                    int width = src.cols;
                    int height = src.rows;
                    uchar *raw = (uchar *)(src.data);
                    Image image(width, height, "Y800", raw, width * height);
                    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
                    int n  = scanner.scan(image); // Scan the Image for Markers
                    cout << "Total Barcodes Scanned :"<<n<<endl;
                    int counter = 0;
                    for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
                        string type=symbol->get_type_name();
                        string name=symbol->get_data() ;
                        // do something useful with results
                        cout    << counter << " "<< "decoded " << type<< " symbol \"" << name << '"' << endl;
                        counter ++;
                    }
                }
                else
                    cout<<"Path entered does not exists!"<<endl;
                break;
            }
            case arucoScan:
            {
                cout<<"Aruco Code Scanning Activated.."<<endl;
                // Insert your code here..
                //Use aruco library from opencv_contrib module
                string path;
                cout<<"Enter Image path :";
                cin>>path;
                if(exists_test(path)){
                    Mat src = imread(path);
                    // Detect Markers in the Image
                    cv::aruco::detectMarkers(src, dictionary, corners, ids);
                    cout << "Total Aruco Scanned :" << ids.size() <<endl;
                    if (ids.size() > 0) // Check if any markers detected and draw them
                        cv::aruco::drawDetectedMarkers(src, corners, ids,Scalar(0,255,0));

                    imshow("Aruco Marker Detected", src);
                    waitKey(0);
                }
                else
                    cout<<"Path entered does not exists!"<<endl;
                break;
            }
            case QRScan:
            {
                cout<<"QR Code Scanning Activated.."<<endl;
                // Insert your code here..
                string path;
                cout<<"Enter Image path :";
                cin>>path;
                if(exists_test(path)){
                    Mat src = imread(path,0);
                    int width = src.cols;
                    int height = src.rows;
                    uchar *raw = (uchar *)(src.data);
                    Image image(width, height, "Y800", raw, width * height);
                    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
                    int n  = scanner.scan(image); // Scan image for Markers
                    cout << "Total Barcodes Scanned :"<<n<<endl;
                    int counter = 0;
                    for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
                        string type=symbol->get_type_name();
                        string name=symbol->get_data() ;
                        // do something useful with results
                        if(type == "QR-Code"){
                            cout    << counter << " "<< "decoded " << type<< " symbol \"" << name << '"' << endl;
                            counter ++;
                        }
                    }
                    cout << "Total QR-Codes scanned :"<<counter<<endl;
                }
                else
                    cout<<"Path entered does not exists!"<<endl;
                break;
            }
            case unknown:
            {
                cout<<cmd<<": command not found"<<endl;
                break;
            }

            }
            // Closes all windows
            destroyAllWindows();
        }
        emit finished();

    }
signals:
    void finished();
};

#endif // LOADTERMINAL_HPP
