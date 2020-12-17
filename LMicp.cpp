//
// Created by echo on 2019/10/14.
//

#include "LMicp.h"
//source 是当前扫描到的点 target 是地图的点
//*************传送大点云用ptr会快很多
Eigen::Isometry3d LMicp::solveICP(pcl::PointCloud<pcl::PointXYZI>::Ptr target, pcl::PointCloud<pcl::PointXYZI> source,
								  Eigen::Isometry3d init_pose) {
	//1.转换点云
	pcl::PointCloud<pcl::PointXYZI> tfed_source;
	pcl::transformPointCloud(source, tfed_source, init_pose.matrix());
	T = Eigen::Isometry3d::Identity();
	//2.计算最近的点云
	pcl::PointCloud<pcl::PointXYZI>::Ptr target_ptr(new pcl::PointCloud<pcl::PointXYZI>);
	target_ptr = target;
	timeCalcSet("kdtreeLast");
	kdtreeLast->setInputCloud(target_ptr); // time 0.0380341 i+15 52% wholemap 1.13772
    timeUsed();
	//进行10次迭代	//3.构建约束 + LM优化 *******wholemap 30 : 0.0135034

	for (int i = 0; i < 70; ++i) {
		//std::cout<<"\n \n current interation: "<<i<<std::endl;
		//solveOneLMNumericDiff(*target,source);
		//solveOneSVD(target_ptr,source);
		//solveOneLM(*target_ptr,source);
		if(solveOneLM(*target_ptr,source)){
			continue;
		} else{
			std::cout<<"times: "<<i<<std::endl;
			break;
		}
	 }

	//4.保存result
/*	pcl::PointCloud<pcl::PointXYZI> result;
	pcl::transformPointCloud(source, result, T.matrix());
	pcl::io::savePCDFile("result.pcd",result); //保存最近点*/
	std::cout<<"Final T\n"<<T.matrix()<<std::endl;
	return T;
}
//1.	用ceres auto diff 求导得到结果
bool LMicp::solveOneLM(pcl::PointCloud<pcl::PointXYZI> target, pcl::PointCloud<pcl::PointXYZI> source) {
	pcl::PointXYZI pointSel; // 要搜索的点
	std::vector<int> pointSearchInd;//搜索最近点的id
	std::vector<float> pointSearchSqDis;//最近点的距离
	ceres::Problem problem;
	ceres::LossFunction *loss_function1 = new ceres::CauchyLoss(0.1); //2.1 设定 loss function
	pcl::PointCloud<pcl::PointXYZI> Temp_pc;
	pcl::transformPointCloud(source, Temp_pc, T.matrix()); //把初始点云转换到上次的结果的位置

	// for (int j = 0; j < 6; ++j) {
	// 	current_psoe[j] = 0;
	// }

	pts2.clear();
	pts1.clear();
	for (int i = 0; i < Temp_pc.size(); ++i) {
		pointSel = Temp_pc[i];
		kdtreeLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
		pts2.push_back(cv::Point3f(pointSel.x,pointSel.y,pointSel.z));
		pts1.push_back(cv::Point3f(target.points[pointSearchInd.back()].x,
								   target.points[pointSearchInd.back()].y,
								   target.points[pointSearchInd.back()].z));
		ceres::CostFunction* cost_function = ICPCeres::Create(pts2[i],pts1[i]); //只有这里用到了自己构造的结构体 在结构体里面确定 error 和 jacobian
		problem.AddResidualBlock(cost_function,
								 loss_function1 /* CauchyLoss loss */,
								 current_psoe);
	}
	
	ceres::Solver::Options options;
	//options.linear_solver_type = ceres::DENSE_SCHUR;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.num_threads=8;
	//options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	
	//std::cout << summary.FullReport() << "\n";
	
	cv::Mat R_vec = (cv::Mat_<double>(3,1) << current_psoe[0],current_psoe[1],current_psoe[2]);//数组转cv向量
	cv::Mat R_cvest;
	Rodrigues(R_vec,R_cvest);//罗德里格斯公式，旋转向量转旋转矩阵
	Eigen::Matrix<double,3,3> R_est;
	cv2eigen(R_cvest,R_est);//cv矩阵转eigen矩阵
	Eigen::Vector3d t_est(current_psoe[3],current_psoe[4],current_psoe[5]);

	Eigen::Isometry3d T_i(R_est);//构造变换矩阵与输出
	T_i.pretranslate(t_est);
	std::cout<<"T increase \n"<<T_i.matrix()<<std::endl;
	T = T*T_i.inverse();	//保存当前的更新
	//std::cout<<"T  \n"<<T.matrix()<<std::endl;
	if (T_i.matrix() == Eigen::Isometry3d::Identity().matrix()){
		return false;
	} else{ return  true;}
}
//2.	运用数值雅克比求得结果
void LMicp::solveOneLMNumericDiff(pcl::PointCloud<pcl::PointXYZI> target, pcl::PointCloud<pcl::PointXYZI> source) {
	pcl::PointXYZI pointSel; // 要搜索的点
	std::vector<int> pointSearchInd;//搜索最近点的id
	std::vector<float> pointSearchSqDis;//最近点的距离
	ceres::Problem problem;
	ceres::LossFunction *loss_function1 = new ceres::CauchyLoss(0.1); //2.1 设定 loss function
	pcl::PointCloud<pcl::PointXYZI> Temp_pc;
	
	pcl::transformPointCloud(source, Temp_pc, T.matrix()); 			//*******把初始点云转换到上次的结果的位置
	for (int j = 0; j < 6; ++j) {
		current_psoe[j] = 0;
	}

	Eigen::Matrix3d information_ = Eigen::Matrix3d::Identity(); //设置information
	for (int i = 0; i < Temp_pc.size(); ++i) {
		pointSel = Temp_pc[i];
		kdtreeLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
	
		vp1 = Eigen::Vector3d(pointSel.x,pointSel.y,pointSel.z);			//当前扫描点
		vp2 = Eigen::Vector3d(target.points[pointSearchInd.back()].x,
								   target.points[pointSearchInd.back()].y,
								   target.points[pointSearchInd.back()].z); //地图点
		//ceres::CostFunction* cost_function = ICPCeres::Create(pts2[i],pts1[i]); //只有这里用到了自己构造的结构体 在结构体里面确定 error 和 jacobian
		ceres::CostFunction * cost_function = new ICPErr(vp1, vp2, information_); //原先设置的是auto-diff
		problem.AddResidualBlock(cost_function,
								 loss_function1 /* CauchyLoss loss */,
								 current_psoe);
	}
	
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Options options1;
	options1.minimizer_type = ceres::TRUST_REGION;
	options1.linear_solver_type = ceres::SPARSE_SCHUR;
	options1.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options1.minimizer_progress_to_stdout = true;
	options1.dogleg_type = ceres::SUBSPACE_DOGLEG;
	ceres::Solver::Summary summary;
	ceres::Solve(options1, &problem, &summary);
	//std::cout << summary.FullReport() << "\n";
	
	cv::Mat R_vec = (cv::Mat_<double>(3,1) << current_psoe[0],current_psoe[1],current_psoe[2]);//数组转cv向量
	cv::Mat R_cvest;
	Rodrigues(R_vec,R_cvest);//罗德里格斯公式，旋转向量转旋转矩阵
	Eigen::Matrix<double,3,3> R_est;
	cv2eigen(R_cvest,R_est);//cv矩阵转eigen矩阵
	Eigen::Vector3d t_est(current_psoe[3],current_psoe[4],current_psoe[5]);
	
	Eigen::Isometry3d T_i(R_est);//构造变换矩阵与输出
	T_i.pretranslate(t_est);
	std::cout<<"T increase \n"<<T_i.matrix()<<std::endl;
	T = T*T_i.inverse();	//保存当前的更新
	std::cout<<"T  \n"<<T.matrix()<<std::endl;
}

bool LMicp::solveOneLMSVD(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R,
						  cv::Mat &t,Eigen::Isometry3d &se3) {
	cv::Point3f p1, p2;     // center of mass
	int N = pts1.size();
	for (int i = 0; i < N; i++) {
		p1 += pts1[i];
		p2 += pts2[i];
	}
	p1 = cv::Point3f(cv::Vec3f(p1) / N);
	p2 = cv::Point3f(cv::Vec3f(p2) / N);
	std::vector<cv::Point3f> q1(N), q2(N); // remove the center
	for (int i = 0; i < N; i++) {
		q1[i] = pts1[i] - p1;
		q2[i] = pts2[i] - p2;
	}
	
	// compute q1*q2^T
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i < N; i++) {
		W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
	}
	Eigen::Matrix3d cc = W.transpose()*W;
	
    Eigen::EigenSolver<Eigen::Matrix3d> es(cc);
	std::cout<<"eigenvalue:\n"<<es.eigenvalues()<<"\n"<<std::endl;
	std::cout<<"eigenvector:\n"<<es.eigenvectors()<<"\n"<<std::endl;
	// SVD on W
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	std::cout << "W=\n" << W << std::endl;
	std::cout << "U=\n" << U << std::endl;
	std::cout << "V=\n" << V << std::endl;
	
	Eigen::Matrix3d R_ = U * (V.transpose());
	if (R_.determinant() < 0) {
		R_ = -R_;
	}
	Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
	
	// convert to cv::Mat
	R = (cv::Mat_<double>(3, 3) <<
							R_(0, 0), R_(0, 1), R_(0, 2),
							R_(1, 0), R_(1, 1), R_(1, 2),
							R_(2, 0), R_(2, 1), R_(2, 2)
	);
	t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
	se3 = Eigen::Isometry3d::Identity();
	se3.rotate(R_);
	se3.pretranslate(t_);
/*	std::cout<<"orign:\n"<<R<<"\n"<<t<<std::endl;
	std::cout<<"result: \n"<<se3.matrix()<<std::endl;*/
	
	return false;
}

void LMicp::solveOneSVD(pcl::PointCloud<pcl::PointXYZI>::Ptr target, pcl::PointCloud<pcl::PointXYZI> source) {
	pcl::PointXYZI pointSel; // 要搜索的点
	std::vector<int> pointSearchInd;//搜索最近点的id
	std::vector<float> pointSearchSqDis;//最近点的距离

	pcl::PointCloud<pcl::PointXYZI> Temp_pc;
	std::vector<cv::Point3f> vp1,vp2;
	pcl::transformPointCloud(source, Temp_pc, T.matrix()); 			//*******把初始点云转换到上次的结果的位置 5.8488e-05
	
	Eigen::Matrix3d information_ = Eigen::Matrix3d::Identity(); //设置information
	
	//for 0.00478317  i+1 65% 的时间 i+5 30% i+15 %14.2 whole map :0.000538211
	//i+15 try2: wholemap 0.000326884 local map:0. 00034949
	//i+1 try2: wholemap  0.00460973 local map:0.00477633
	//用了94%的时间
	timeCalcSet("for");
	for (int i = 0; i < Temp_pc.size(); i = i+1) {
	
		pointSel = Temp_pc[i];
		kdtreeLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); //	1.294e-06

		vp1.push_back(cv::Point3f(pointSel.x,pointSel.y,pointSel.z)) ;			//当前扫描点 8.2e-08
		vp2.push_back(cv::Point3f(target->points[pointSearchInd.back()].x,
								  target->points[pointSearchInd.back()].y,
								  target->points[pointSearchInd.back()].z)) ; //地图点

	}
	timeUsed();
	cv::Mat R, t;
	Eigen::Isometry3d se3;

	solveOneLMSVD(vp1,vp2,R,t,se3); //4.865e-06
	std::cout<<se3.matrix()<<std::endl;
	T = T*se3.inverse();

}


//*****带jacobian 的icp 计算部分
LMicp::ICPErr::ICPErr(Eigen::Vector3d &pi, Eigen::Vector3d &pj, Eigen::Matrix<double, 3, 3> &information)
	:Pi(pi), Pj(pj) {
	Eigen::LLT<Eigen::Matrix<double, 3, 3>> llt(information);
	sqrt_information_ = llt.matrixL();
}

bool LMicp::ICPErr::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
	Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(parameters[0]);
	Sophus::SE3d T = Sophus::SE3d::exp(lie); //指数映射到se3
	
	//std::cout << T.matrix3x4() << std::endl;
	
	auto Pj_ = T * Pi; //SE3 4*4 可以直接乘 Vector3d ???
	Eigen::Vector3d err = Pj - Pj_; //向量相减
	
	//err = sqrt_information_ * err; 	//信息矩阵,相当于给每一个配了不同的权重
	
	residuals[0] = err(0);
	residuals[1] = err(1);
	residuals[2] = err(2);
	
	Eigen::Matrix<double, 3, 6> Jac = Eigen::Matrix<double, 3, 6>::Zero();
	Jac.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
	Jac.block<3, 3>(0, 3) = Sophus::SO3d::hat(Pj_);// hat 为向量到反对称矩阵  相对的，vee为反对称到向量
	int k = 0;
	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 6; ++j) {
			if(k >= 18)
				return false;
			
			if(jacobians) {
				if(jacobians[0])
					jacobians[0][k] = Jac(i, j);//0-18
			}
			k++;
		}
	}
	return true;
}
