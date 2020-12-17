//
// Created by echo on 2019/10/14.
//
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <chrono>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include<ceres/ceres.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "data/rotation.h"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

#ifndef LMICP_LMICP_H
#define LMICP_LMICP_H


class LMicp {
public:
	LMicp(){kdtreeLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());};
	Eigen::Isometry3d solveICP(pcl::PointCloud<pcl::PointXYZI>::Ptr target, pcl::PointCloud<pcl::PointXYZI> source,
							   Eigen::Isometry3d init_pose) ;
	//1.	用ceres auto diff 求导得到结果
	bool solveOneLM(pcl::PointCloud<pcl::PointXYZI> target, pcl::PointCloud<pcl::PointXYZI> source);
	//2.	运用数值雅克比求得结果
	void solveOneLMNumericDiff(pcl::PointCloud<pcl::PointXYZI> target, pcl::PointCloud<pcl::PointXYZI> source);
	//3.	SVD 分解得到结果
	bool solveOneLMSVD (const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2,
						cv::Mat &R, cv::Mat &t, Eigen::Isometry3d &se3);
	void solveOneSVD(pcl::PointCloud<pcl::PointXYZI>::Ptr target, pcl::PointCloud<pcl::PointXYZI> source);
	//4. 	手动求解L-M
	Eigen::Isometry3d solveICPHand(pcl::PointCloud<pcl::PointXYZI>::Ptr target, pcl::PointCloud<pcl::PointXYZI> source,
							   Eigen::Isometry3d init_pose) ;

	
	void timeUsed() {
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
				(std::chrono::high_resolution_clock::now() - _now_ms);
		std::cout<<name_global<<" time is :"<< duration.count()/1e9<<std::endl;
		
	}
	
	void timeCalcSet(std::string name) {
		_now_ms = std::chrono::high_resolution_clock ::now();
		name_global = name;
		
	}
	
    double current_psoe[6]={0,0,0,0,0,0};
private:
	std::string name_global;
	std::chrono::high_resolution_clock::time_point _now_ms;
	Eigen::Isometry3d T;
	std::vector<cv::Point3f> pts1, pts2; //目前的点对 1:目标点 2.调整点
	Eigen::Vector3d vp1,vp2;
	pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeLast;
	
	//@ 1.ceres 通过重载()来计算残差
	struct ICPCeres
	{
		//其中，形参uvw是第一帧的相机观测到的坐标，xyz是第二帧的相机观测到的坐标，由于仅仅优化位姿，
		// 我们假设第二帧相机坐标确定，AngleAxisRotatePoint在头文件rotation.h中，
		// 它根据相机位姿（旋转向量和平移向量表示，构成的6维数组，不对内参焦距进行优化，不考虑相机畸变），
		// 第二帧的相机位姿（三维数组），计算得到RP，结合平移向量得到计算后第一帧相机坐标，进而计算误差e=p-（Rp‘+t）。
		ICPCeres ( cv::Point3f uvw,cv::Point3f xyz ) : _uvw(uvw),_xyz(xyz) {}
		// 残差的计算
		template <typename T>
		bool operator() (
				const T* const camera,     // 位姿参数，有6维
				T* residual ) const        // 残差
		{
			//xyz 为
			T p[3];
			T point[3];
			point[0]=T(_xyz.x);
			point[1]=T(_xyz.y);
			point[2]=T(_xyz.z);
			AngleAxisRotatePoint(camera, point, p);//计算RP 输入point 和rotate camera 输出旋转过的坐标
			p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];//新特征点的坐标 3,4,5 为 t
			residual[0] = T(_uvw.x)-p[0]; //x方向的error
			residual[1] = T(_uvw.y)-p[1]; //y方向的error
			residual[2] = T(_uvw.z)-p[2]; //z方向的error
			return true;
		}
		//todo: 改成数值雅克比?
		static ceres::CostFunction* Create(const cv::Point3f uvw,const cv::Point3f xyz) {
			return (new ceres::AutoDiffCostFunction<ICPCeres, 3, 6> (new ICPCeres(uvw,xyz) ) );
		}

		const cv::Point3f _uvw;
		const cv::Point3f _xyz;
	};
	
	//@ 2.定义一下icp的error function
	class ICPErr : public ceres::SizedCostFunction<3, 6> {
	public:
		ICPErr(Eigen::Vector3d& pi, Eigen::Vector3d &pj,
			   Eigen::Matrix<double, 3, 3> &information);
		virtual ~ICPErr() {}
		virtual bool Evaluate(double const* const* parameters,
							  double* residuals,
							  double** jacobians) const;
	
	public:
		Eigen::Vector3d Pi;
		Eigen::Vector3d Pj;
		Eigen::Matrix<double, 3, 3> sqrt_information_;
	};
	
};


#endif //LMICP_LMICP_H
