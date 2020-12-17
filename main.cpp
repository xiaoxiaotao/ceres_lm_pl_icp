#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "LMicp.h"
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/common/transforms.h>


int main() {
	//整个地图
	std::string _whole_map = "/home/tao/hdl_slam_pcd_ws/map3/000022/cloud.pcd";
	//当前扫描的点云
	std::string cloud_frame = "/home/tao/hdl_slam_pcd_ws/map3/000023/cloud.pcd";
	//设置初始的位姿





	Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
	pcl::PointCloud<pcl::PointXYZI>target_cloud;
	pcl::PointCloud<pcl::PointXYZI> source;
	pcl::PointCloud<pcl::PointXYZI> output_cloud;

	pcl::PointCloud<pcl::PointXYZI>::Ptr fast_map(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr source_(new pcl::PointCloud<pcl::PointXYZI>);

	pcl::io::loadPCDFile<pcl::PointXYZI>(_whole_map, *fast_map);
	pcl::io::loadPCDFile<pcl::PointXYZI>(cloud_frame, source);
	pcl::io::loadPCDFile<pcl::PointXYZI>(cloud_frame, *source_);




 	float theta1= M_PI/8;
    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
    transform_2.translation() << 1.5,0.5, 0;
    transform_2.rotate (Eigen::AngleAxisf (theta1, Eigen::Vector3f::UnitZ()));
    std::cout <<"transform matrix: "<< transform_2.matrix() << std::endl;

    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZI> ());
    pcl::transformPointCloud (source, *transformed_cloud, transform_2);




	std::cout << "--- pcl::ICP ---" << std::endl;
	pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp2; //创建ICP对象，用于ICP配准
	icp2.setMaximumIterations(500);    //设置最大迭代次数iterations=true
	icp2.setInputSource(source_); //设置输入点云
	icp2.setInputTarget(transformed_cloud); //设置目标点云（输入点云进行仿射变换，得到目标点云）
	icp2.align(output_cloud);          //匹配后源点云
    std::cout<<icp2.getFitnessScore()<<std::endl;
    std::cout<<icp2.getFinalTransformation()<<std::endl;
    std::cout<<icp2.hasConverged()<<std::endl;
    std::cout<<icp2.nr_iterations_ <<std::endl;


	LMicp lm,a;
	a.timeCalcSet("lm");
	//lm.solveOneLM(fast_map,source,raw_pose);
	lm.solveICP(transformed_cloud,*source_,raw_pose);
	a.timeUsed();
	return 0;
}