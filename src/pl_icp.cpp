#include "pl_icp.h"


bool PL_ICP::transform_update(){

    pcl::PointXYZI pointSel; // 要搜索的点
	std::vector<int> pointSearchInd;//搜索最近点的id
	std::vector<float> pointSearchSqDis;//最近点的距离
	ceres::Problem problem;
	ceres::LossFunction *loss_function1 = new ceres::CauchyLoss(0.1); //2.1 设定 loss function
	pcl::PointCloud<pcl::PointXYZI> Temp_pc;
	pcl::transformPointCloud(source_cloud, Temp_pc, T.matrix()); //把初始点云转换到上次的结果的位置
	Eigen::Isometry3d T_pre=T;

#pragma omp parallel for
	for (int j = 0; j < 6; ++j) {
		pose[j] = 0;
	}
	for (int i = 0; i < Temp_pc.size(); ++i) {
		pointSel = Temp_pc[i];
		kdtreeLast.nearestKSearch(pointSel, 2, pointSearchInd, pointSearchSqDis);
		Eigen::Vector3d p(pointSel.x,pointSel.y,pointSel.z);
		Eigen::Vector3d p1(target_cloud[pointSearchInd.back()].x,
					   target_cloud[pointSearchInd.back()].y,
					   target_cloud[pointSearchInd.back()].z);
		Eigen::Vector3d p2(target_cloud[pointSearchInd.front()].x,
					   target_cloud[pointSearchInd.front()].y,
					   target_cloud[pointSearchInd.front()].z);	

		ceres::CostFunction* cost_function = ICP_Line_Ceres::Create(p,p1,p2);
		problem.AddResidualBlock(cost_function,
								 loss_function1 /* CauchyLoss loss */,
								 pose);
	}



	ceres::Solver::Options options;

	// ceres::TinySolver::Options options_tiny;
	//DENSE_NORMAL_CHOLESKY
	//SPARSE_NORMAL_CHOLESKY
	options.minimizer_type = ceres::TRUST_REGION;
  	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type =  ceres::DENSE_NORMAL_CHOLESKY;
	// options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 100;
	options.num_threads=8;
	// options.gradient_tolerance = 1e-6;
	// options.function_tolerance = 1e-6;
	

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

    double qq[4];
    double angle[3];
    angle[0]=pose[0];
    angle[1]=pose[1];
    angle[2]=pose[2];
    ceres::AngleAxisToQuaternion(angle, qq);
    dt = Eigen::Vector3d(pose[3], pose[4], pose[5]);
    dq.w() = qq[0];
    dq.x() = qq[1];
    dq.y() = qq[2];
    dq.z() = qq[3];
    AngleAxisd rotation_vector(dq);
    Eigen::Matrix3d R;
    R=dq.matrix();
    Eigen::Isometry3d T_i(R);
   
    T_i.rotate(rotation_vector);                  
    T_i.pretranslate(dt); 
    T=T*T_i.inverse();
    std::cout<<T.matrix()<<std::endl;

 	double delta_T= compute_delta_T(T_pre,T);
	//  std::cout<<delta_T<<std::endl;
	if(delta_T<=0.0005){
		return false;
	}else{
		return true;
	}

	if (T_i.matrix() == Eigen::Isometry3d::Identity().matrix()){
		return false;
	} else{ return  true;}




	  
}

void PL_ICP::lmsetInputSource(pcl::PointCloud<pcl::PointXYZI> source_clouds){

	source_cloud=source_clouds;
}
	
void PL_ICP::lmsetInputTarget(pcl::PointCloud<pcl::PointXYZI> target_clouds){

	target_cloud=target_clouds;

}

void PL_ICP::pl_icp(Eigen::Isometry3d T_init){

    setupInitParams();    
    T = T_init;
    for(int i=0;i< 100;i++){
        if(transform_update()){
            std::cout<<i<<std::endl;
        }else{
            break;
        }
    }
   
    std::cout<<"tranform T: "<<T.matrix()<<std::endl;


}

void PL_ICP::setupInitParams(){

    kdtreeLast.setInputCloud(target_cloud.makeShared());
	q = Eigen::Quaterniond::Identity();
    t = Eigen::Vector3d::Zero();
    dq = Eigen::Quaterniond::Identity();
    dt = Eigen::Vector3d::Zero();

}

double PL_ICP::compute_delta_T(Eigen::Isometry3d t1, Eigen::Isometry3d t2){

	double error_sum[16];
	double delta_T=0;

#pragma omp parallel for
	for(int i=0;i<4;i++){

		error_sum[i*4+0]=std::abs(t2(i,0)-t1(i,0));
		error_sum[i*4+1]=std::abs(t2(i,1)-t1(i,1));
		error_sum[i*4+2]=std::abs(t2(i,2)-t1(i,2));
		error_sum[i*4+3]=std::abs(t2(i,3)-t1(i,3));

	}

	for(int i=0;i<16;i++){
		delta_T+=error_sum[i];

	}

	return delta_T/16;


}
