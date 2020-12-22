#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "pl_icp.h"
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h> 
#include <pcl/console/time.h>   // TicToc



// ICP（Iterative Closest Point），即最近点迭代算法，是最为经典的数据配准算法。
// 其特征在于，通过求取源点云和目标点云之间的对应点对，基于对应点对构造旋转平移矩阵，
// 并利用所求矩阵，将源点云变换到目标点云的坐标系下，估计变换后源点云与目标点云的误差函数，
// 若误差函数值大于阀值，则迭代进行上述运算直到满足给定的误差要求.




int main() {
	//整个地图
	std::string target_path = "/home/tao/hdl_slam_pcd_ws/map3/000008/cloud.pcd";
	//当前扫描的点云
	std::string source_path = "/home/tao/hdl_slam_pcd_ws/map3/000009/cloud.pcd";
	//设置初始的位姿

	Eigen::Isometry3d raw_pose = Eigen::Isometry3d::Identity();
	// pcl::PointCloud<pcl::PointXYZI>target_cloud;
	// pcl::PointCloud<pcl::PointXYZI> source;
	pcl::PointCloud<pcl::PointXYZI> output_cloud;

	pcl::PointCloud<pcl::PointXYZI>::Ptr target(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>);

	pcl::io::loadPCDFile<pcl::PointXYZI>(target_path, *target);
	pcl::io::loadPCDFile<pcl::PointXYZI>(source_path, *source);

	pcl::VoxelGrid<pcl::PointXYZI> filter;
	filter.setInputCloud(source);
	filter.setLeafSize(0.25f, 0.25f, 0.25f);
	filter.filter(*source);
	


	filter.setInputCloud(target);
	filter.setLeafSize(0.25f, 0.25f, 0.25f);
	filter.filter(*target);


	pcl::ConditionAnd<pcl::PointXYZI>::Ptr condi_cloud(new pcl::ConditionAnd<pcl::PointXYZI> ());
	condi_cloud->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI>("z", pcl::ComparisonOps::GT, 0.15)));
	pcl::ConditionalRemoval<pcl::PointXYZI> condream;
	
	condream.setCondition(condi_cloud);
	condream.setInputCloud(source);
	condream.filter(*source);

	condream.setCondition(condi_cloud);
	condream.setInputCloud(target);
	condream.filter(*target);


	std::cout<<"filter size:"<<target->size()<<std::endl;
	std::cout<<"filter size:"<<source->size()<<std::endl;




	// //M_PI/16
 	// float theta1= M_PI/16;
    // Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
    // transform_2.translation() << 1.5,1, 0;
    // transform_2.rotate (Eigen::AngleAxisf (theta1, Eigen::Vector3f::UnitZ()));
    // std::cout <<"transform matrix: "<< transform_2.matrix() << std::endl;

    // pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZI> ());
    //pcl::transformPointCloud (*source, *target, transform_2);




	std::cout << "--- pcl::ICP ---" << std::endl;
	pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp2; //创建ICP对象，用于ICP配准
	icp2.setMaximumIterations(500);    //设置最大迭代次数iterations=true
	icp2.setInputSource(source); //设置输入点云
	icp2.setInputTarget(target); //设置目标点云（输入点云进行仿射变换，得到目标点云）
	icp2.align(output_cloud);          //匹配后源点云
	
    std::cout<<icp2.getFitnessScore()<<std::endl;
    std::cout<<icp2.getFinalTransformation()<<std::endl;
    std::cout<<icp2.hasConverged()<<std::endl;
    std::cout<<icp2.nr_iterations_ <<std::endl;

	Eigen::Isometry3d T_init;

	for(int i=0;i<4;i++){

		T_init(i,0) = icp2.getFinalTransformation()(i,0);
		T_init(i,1) = icp2.getFinalTransformation()(i,1);
		T_init(i,2) = icp2.getFinalTransformation()(i,2);
		T_init(i,3) = icp2.getFinalTransformation()(i,3);


	}

	PL_ICP plicp;// pl icp 需要较好的初值，否则会陷入局部最优
	plicp.lmsetInputSource(*source);
	plicp.lmsetInputTarget(*target);
	plicp.T=T_init;
	std::cout<<"init T: "<<plicp.T.matrix()<<std::endl;
	pcl::console::TicToc time;
	time.tic();

	plicp.pl_icp(T_init);
	std::cout <<  " pl icp use time: " << time.toc() << " ms" << std::endl;

	return 0;
}