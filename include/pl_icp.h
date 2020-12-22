#ifndef LMICP_LMICP_H
#define LMICP_LMICP_H

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <chrono>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <omp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/approximate_voxel_grid.h> 
#include <ceres/tiny_solver.h>
#include <ceres/tiny_solver_autodiff_function.h>
#include <ceres/tiny_solver_cost_function_adapter.h>


using namespace Eigen;
using namespace std;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using ceres::AngleAxisRotatePoint;
using ceres::CrossProduct;
using ceres::DotProduct;



class PL_ICP {
public:

	PL_ICP(){};

	bool transform_update();

    void transform_accumulate();

	double compute_delta_T(Eigen::Isometry3d t1, Eigen::Isometry3d t2);

	void lmsetInputSource(pcl::PointCloud<pcl::PointXYZI> source_cloud);
	
	void lmsetInputTarget(pcl::PointCloud<pcl::PointXYZI> target_cloud);

	void setupInitParams();

	void pl_icp(Eigen::Isometry3d T_init);

	double pose[6]={0,0,0,0,0,0};

	double error=0;

	// double delta_T;

	Eigen::Isometry3d T;
	Eigen::Quaterniond q;
    Eigen::Vector3d t;
    Eigen::Quaterniond dq;
    Eigen::Vector3d dt;

	pcl::KdTreeFLANN<pcl::PointXYZI>  kdtreeLast;
	pcl::PointCloud<pcl::PointXYZI> source_cloud;
	pcl::PointCloud<pcl::PointXYZI> target_cloud;

	struct ICP_Line_Ceres
		{
		
		ICP_Line_Ceres ( Eigen::Vector3d p, Eigen::Vector3d p1,Eigen::Vector3d p2 ) : p(p),p1(p1),p2(p2) {}
		// 残差的计算
		template <typename T>
		bool operator() (
				const T* const pose,     // 位姿参数，有6维
				T* residual ) const        // 残差
		{
			
			T pi[3];
			T p1_[3];
			T p2_[3];
			pi[0] = T(p(0));
			pi[1] = T(p(1));
			pi[2] = T(p(2));

			p1_[0] = T(p1(0));
			p1_[1] = T(p1(1));
			p1_[2] = T(p1(2));

			p2_[0] = T(p2(0));
			p2_[1] = T(p2(1));
			p2_[2] = T(p2(2));


			T pi_proj_1[3];//project pi to current frame
			T pi_proj_2[3];
			AngleAxisRotatePoint(pose, p1_, pi_proj_1);
			AngleAxisRotatePoint(pose, p2_, pi_proj_2);

			pi_proj_1[0] += pose[3];
			pi_proj_1[1] += pose[4];
			pi_proj_1[2] += pose[5];

			pi_proj_2[0] += pose[3];
			pi_proj_2[1] += pose[4];
			pi_proj_2[2] += pose[5];


			// //distance between pi to line(pi_proj_1, pi_proj_2)
			T d1[3], d2[3], d12[3];
			d1[0] = pi[0] - pi_proj_1[0];
			d1[1] = pi[1] - pi_proj_1[1];
			d1[2] = pi[2] - pi_proj_1[2];

			d2[0] = pi[0] - pi_proj_2[0];
			d2[1] = pi[1] - pi_proj_2[1];
			d2[2] = pi[2] - pi_proj_2[2];

			d12[0] = pi_proj_1[0]- pi_proj_2[0];
			d12[1] = pi_proj_1[1]- pi_proj_2[1];
			d12[2] = pi_proj_1[2]- pi_proj_2[2];

			T cross[3];
			CrossProduct(d1, d2, cross);

			//h=||(a * b)}} / ||z||

			T norm = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
			T norm12 = sqrt(d12[0] * d12[0] + d12[1] * d12[1] + d12[2] * d12[2]);
			T weight = T(10.0);
			residual[0]= weight * norm / norm12;

		
			return true;
		}

		static ceres::CostFunction* Create(Eigen::Vector3d p,const Eigen::Vector3d p1,const Eigen::Vector3d p2) {
			return (new ceres::AutoDiffCostFunction<ICP_Line_Ceres, 1, 6> (new ICP_Line_Ceres(p,p1,p2)));
			}	

	// static ceres::CostFunction* Create(Eigen::Vector3d p,const Eigen::Vector3d p1,const Eigen::Vector3d p2) {
	// 		return (new ceres::TinySolverAutoDiffFunction<ICP_Line_Ceres, 1, 6> (new ICP_Line_Ceres(p,p1,p2)));
	// 		}	





		const Eigen::Vector3d p;
		const Eigen::Vector3d p1;
		const Eigen::Vector3d p2;
	};


};
#endif //LMICP_LMICP_H
