package com.example.nativethreadtest;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import android.os.Bundle;
import android.support.v7.app.ActionBarActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.CheckBox;

public class MainActivity extends ActionBarActivity implements
		CvCameraViewListener2 {
	private CameraBridgeViewBase mOpenCvCameraView;
	private Mat currentImage;
	private CheckBox threadCheck,ratioCheck;
	
	public native void capturePattern(long address);
	public native void updateImage(long address);
	public native void singleThread();
	public native void MultiThread();
	public native void getOutPutImage(long address);
	public native int check1();
	public native int check2();
	public native void initializeSystem();
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		System.loadLibrary("threaded");
		setContentView(R.layout.activity_main);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.javaCameraView1);
		mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
		mOpenCvCameraView.setMaxFrameSize(720, 480);
		mOpenCvCameraView.setCvCameraViewListener(this);
		threadCheck = (CheckBox)findViewById(R.id.checkBox2);
		ratioCheck = (CheckBox)findViewById(R.id.checkBox1);
		initializeSystem();
	}
	
	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// Handle action bar item clicks here. The action bar will
		// automatically handle clicks on the Home/Up button, so long
		// as you specify a parent activity in AndroidManifest.xml.
		int id = item.getItemId();
		if (id == R.id.action_settings) {
			return true;
		}
		return super.onOptionsItemSelected(item);
	}

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i("OpenCV", "OpenCV loaded successfully");
				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	@Override
	public void onCameraViewStarted(int width, int height) {
		// TODO Auto-generated method stub

	}

	@Override
	public void onCameraViewStopped() {
		// TODO Auto-generated method stub

	}

	@Override
	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		currentImage = inputFrame.rgba();
		updateImage(currentImage.getNativeObjAddr());
		if(!threadCheck.isChecked()){
			singleThread();
		}else{
			MultiThread();
		}
		getOutPutImage(currentImage.getNativeObjAddr());
		return currentImage;
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this,
				mLoaderCallback);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}
	
	//UI action 
	public void setPattern(View view){
		Log.e("Main Activity", "Pattern Set");
		capturePattern(currentImage.getNativeObjAddr());
	}
}
