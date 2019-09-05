#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <string>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/board.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp.h>

#include <pcl/registration/correspondence_rejection_one_to_one.h>
using namespace std;
pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud, float leaf_size) {
	//体素滤波
	pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;  //创建滤波对象
	sor.setInputCloud(Acloud);            //设置需要过滤的点云给滤波对象
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);  //设置滤波时创建的体素体积
	sor.filter(*Acloud_filtered);           //执行滤波处理，存储输出	
	return Acloud_filtered;
}

pcl::PointCloud<pcl::Normal>::Ptr normal_estimation_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNumberOfThreads(10);
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(5.0f*leaf_size);
	ne.compute(*normals);
	return normals;
}

bool is_max(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<float> angle, int i, vector<bool>& possible_key, pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree, float leaf_size = 5) {
	vector<int> point_ind;
	vector<float> point_dis;
	kdtree->radiusSearch(cloud->points[i], leaf_size, point_ind, point_dis);

	//if (point_ind.size() < 20)
	//	return false;


	for (int i = 1; i < point_ind.size(); i++) {
		if (angle[point_ind[0]] >= angle[point_ind[i]])
			possible_key[point_ind[i]] = false;
		else if (angle[point_ind[0]] < angle[point_ind[i]])
			possible_key[point_ind[0]] = false;

	}
	return possible_key[point_ind[0]];

}

//float com_angle(float cx, float cy, float cz, float nx, float ny, float nz) {
//	if (isnan(nx) || isnan(ny) || isnan(nz) || isnan(cx) || isnan(cy) || isnan(cz) || (cx == nx && cy == ny && cz == nz))
//		return 0;
//	float cos_angle = (nx*cx + ny * cy + nz * cz) / (sqrtf(pow(nx, 2) + pow(ny, 2) + pow(nz, 2))*sqrtf(pow(cx, 2) + pow(cy, 2) + pow(cz, 2)));
//	if (cos_angle < 0)
//		return 90.0f + acos(cos_angle)*180.0f / 3.1415926;
//
//	return acos(cos_angle)*180.0f / 3.1415926;
//}

pcl::PointCloud<pcl::PointXYZ>::Ptr key_detect(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree, float leaf_size, vector<float>& l) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	vector<bool> possible_key(cloud->size(), true);

	vector<float> angle;

	for (int i = 0; i < cloud->size(); i++) {
		if (normal->points[i].curvature > 0.02) {
			std::vector<int> id;//最近点索引
			std::vector<float> dis;//最近点距离
			kdtree->radiusSearch(cloud->points[i], leaf_size, id, dis);
			if (id.size() < 50) {
				angle.push_back(0);
				continue;
			}
			pcl::Normal avg_normal;
			for (int j = 0; j < id.size(); j++) {
				//cos_angle += com_angle(normal->points[i].normal_x, normal->points[i].normal_y, normal->points[i].normal_z,
				//	normal->points[id[j]].normal_x, normal->points[id[j]].normal_y, normal->points[id[j]].normal_z);
				avg_normal.normal_x += normal->points[id[j]].normal_x;
				avg_normal.normal_y += normal->points[id[j]].normal_y;
				avg_normal.normal_z += normal->points[id[j]].normal_z;
			}
			avg_normal.normal_x = avg_normal.normal_x / float(id.size());
			avg_normal.normal_y = avg_normal.normal_y / float(id.size());
			avg_normal.normal_z = avg_normal.normal_z / float(id.size());
			float cos_angle = 0;
			for (int j = 0; j < id.size(); j++) {
				//cos_angle += com_angle(normal->points[i].normal_x, normal->points[i].normal_y, normal->points[i].normal_z,
				//	normal->points[id[j]].normal_x, normal->points[id[j]].normal_y, normal->points[id[j]].normal_z);

				cos_angle += pow(normal->points[i].normal_x - avg_normal.normal_x, 2) +
					pow(normal->points[i].normal_y - avg_normal.normal_y, 2) +
					pow(normal->points[i].normal_z - avg_normal.normal_z, 2);
			}
			cos_angle = cos_angle / (float)id.size();
			angle.push_back(cos_angle);

		}
		else {
			angle.push_back(0);
			possible_key[i] = false;
		}
	}

	for (int i = 0; i < cloud->size(); i++) {
		if (possible_key[i]) {
			if (is_max(cloud, angle, i, possible_key, kdtree, leaf_size)&&angle[i]!=0) {//此处半径为计算曲率和的半径
				key->push_back(cloud->points[i]);
				l.push_back(angle[i]);
			}
		}
	}


	return key;
}

void finall_key(pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, vector<float> ls, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, vector<float> lt) {
	if (ls.size() < 300 || lt.size() < 300)
		return;
	vector<float> ls_order = ls;
	vector<float> lt_order = lt;
	sort(ls_order.begin(), ls_order.end());
	sort(lt_order.begin(), lt_order.end());
	float lsmin, lsmax, ltmin, ltmax, lmin, lmax;
	lsmax = ls_order[ls.size() - 1];
	ltmax = lt_order[lt.size() - 1];
	lsmin = ls_order[ls.size() - 300];
	ltmin = lt_order[lt.size() - 300];
	if (lsmax< ltmin || lsmin>ltmax)
		return;
	lmin = max(lsmin, ltmin);
	lmax = min(lsmax, ltmax);

	pcl::PointCloud<pcl::PointXYZ>::Ptr finall_key_source(new pcl::PointCloud<pcl::PointXYZ>);
	*finall_key_source = *key_source;
	key_source->clear();
	pcl::PointCloud<pcl::PointXYZ>::Ptr finall_key_target(new pcl::PointCloud<pcl::PointXYZ>);
	*finall_key_target = *key_target;
	key_target->clear();

	for (int i = 0; i < finall_key_source->size(); i++) {
		if (ls[i] >= lmin && ls[i] <= lmax) {
			key_source->push_back(finall_key_source->points[i]);
		}
	}
	for (int i = 0; i < finall_key_target->size(); i++) {
		if (lt[i] >= lmin && lt[i] <= lmax) {
			key_target->push_back(finall_key_target->points[i]);
		}
	}
	return;
}

void show_key_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
	viewer->setBackgroundColor(255, 255, 255, v1);    //设置视口的背景颜色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_source(key_source, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(key_source, color_key_source, "key_source", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_source");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_source(cloud_source, 0, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_source, color_cloud_source, "cloud_model", v1);

	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
	viewer->setBackgroundColor(255, 255, 255, v2);    //设置视口的背景颜色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_target(key_target, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(key_target, color_key_target, "key_target", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_target");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_target(cloud_target, 0, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_target, color_cloud_target, "cloud_target", v2);

	// 等待直到可视化窗口关闭
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		//boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_key_point(pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypoint) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);  //白色背景

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(Acloud, 0, 255, 0);//蓝色点云
	viewer_final.addPointCloud<pcl::PointXYZ>(Acloud, color_cloud, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key(cloud_keypoint, 255, 0, 0);//关键点
	viewer_final.addPointCloud<pcl::PointXYZ>(cloud_keypoint, color_key, "2");
	viewer_final.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "2");

	// 等待直到可视化窗口关闭
	while (!viewer_final.wasStopped())
	{
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_one_key_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr before_trans_key, pcl::PointCloud<pcl::PointXYZ>::Ptr after_trans_key, pcl::PointCloud<pcl::PointXYZ>::Ptr id) {
	pcl::visualization::PCLVisualizer line("line");


	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud, 0.0, 0.0, 255.0), "cloud");
	line.addPointCloud<pcl::PointXYZ>(before_trans_key, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(before_trans_key, 255, 0, 0), "before_trans_key");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "before_trans_key");
	line.addPointCloud<pcl::PointXYZ>(after_trans_key, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(after_trans_key, 255, 0, 0), "after_trans_key");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "after_trans_key");

	line.addPointCloud<pcl::PointXYZ>(id, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(id, 0, 255, 0), "id");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "id");

	while (!line.wasStopped())
	{
		line.spinOnce();
	}

}

float com_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree) {
	//进行1邻域点搜索
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//最近点索引
	std::vector<float> pointNKNSquaredDistance(K);//最近点距离
	//在B点云中计算点与最近邻的平均距离
	float leaf_size = 0;
	for (int i = 0; i < cloud->size(); i++) {
		kdtree->nearestKSearch(i, K, pointIdxNKNSearch, pointNKNSquaredDistance);
		leaf_size = leaf_size + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	leaf_size = (float)leaf_size / (float)(cloud->size());
	return leaf_size;
}

pcl::PointCloud<pcl::PFHSignature125>::Ptr com_features(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr key, pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree, pcl::PointCloud<pcl::ReferenceFrame>& rf_source, float leaf_size, float m = 5.0f) {

	//pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf_source(new pcl::PointCloud<pcl::ReferenceFrame>());
	pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> rf_est;
	rf_est.setFindHoles(true);
	rf_est.setRadiusSearch(5.0f*leaf_size);
	rf_est.setInputCloud(key);
	rf_est.setInputNormals(normal);
	rf_est.setSearchSurface(cloud);
	rf_est.compute(rf_source);

	pcl::PointCloud<pcl::PFHSignature125>::Ptr features(new pcl::PointCloud<pcl::PFHSignature125>);
	for (int i = 0; i < key->size(); i++) {

		pcl::PFHSignature125 feature;
		for (int i = 0; i < feature.descriptorSize(); i++) {
			feature.histogram[i] = 0;
		}

		if (isnan(rf_source.points[i].x_axis[0])) {
			features->push_back(feature);
			continue;
		}

		std::vector<int> id;//最近点索引
		std::vector<float> dis;//最近点距离
		kdtree->radiusSearch(key->points[i], m*2.5f*sqrt(3.0f)*leaf_size, id, dis);
		int num = 0;
		//pcl::PointCloud<pcl::PointXYZ>::Ptr before_trans_key_source(new pcl::PointCloud<pcl::PointXYZ>);
		//pcl::PointCloud<pcl::PointXYZ>::Ptr after_trans_key_source(new pcl::PointCloud<pcl::PointXYZ>);
		//pcl::PointCloud<pcl::PointXYZ>::Ptr key_id(new pcl::PointCloud<pcl::PointXYZ>);
		//key_id->push_back(key->points[i]);
		for (int j = 0; j < id.size(); j++) {

			pcl::PointXYZ point0;
			point0.x = cloud->points[id[j]].x - key->points[i].x;
			point0.y = cloud->points[id[j]].y - key->points[i].y;
			point0.z = cloud->points[id[j]].z - key->points[i].z;
			pcl::PointXYZ point;
			point.x = rf_source.points[i].x_axis[0] * point0.x
				+ rf_source.points[i].y_axis[0] * point0.y
				+ rf_source.points[i].z_axis[0] * point0.z;
			if (abs(point.x) >= m * 2.5f*leaf_size)
				continue;
			point.y = rf_source.points[i].x_axis[1] * point0.x
				+ rf_source.points[i].y_axis[1] * point0.y
				+ rf_source.points[i].z_axis[1] * point0.z;
			if (abs(point.y) >= m * 2.5f*leaf_size)
				continue;
			point.z = rf_source.points[i].x_axis[2] * point0.x
				+ rf_source.points[i].y_axis[2] * point0.y
				+ rf_source.points[i].z_axis[2] * point0.z;
			if (abs(point.z) >= m * 2.5f*leaf_size)
				continue;
			int x, y, z = 0;
			num += 1;
			x = (point.x + m * 2.5f* leaf_size) / (m*leaf_size);
			y = (point.y + m * 2.5f* leaf_size) / (m*leaf_size);
			z = (point.z + m * 2.5f* leaf_size) / (m*leaf_size);
			feature.histogram[x + 5 * y + 25 * z] += 1;
			//before_trans_key_source->push_back(cloud->points[id[j]]);
			//after_trans_key_source->push_back(point);
		}
		//cout << rf_source->points[i].x_axis[0]<<"    " << rf_source->points[i].x_axis[1] << "    "<< rf_source->points[i].x_axis[2] << endl;
		//cout << rf_source->points[i].y_axis[0] << "    " << rf_source->points[i].y_axis[1] << "    " << rf_source->points[i].y_axis[2] << endl;
		//cout << rf_source->points[i].z_axis[0] << "    " << rf_source->points[i].z_axis[1] << "    " << rf_source->points[i].z_axis[2] << endl;

		//show_one_key_cloud(cloud, before_trans_key_source, after_trans_key_source,key_id);
		for (int k = 0; k < feature.descriptorSize(); k++) {
			feature.histogram[k] = feature.histogram[k] / float(num);
		}
		features->push_back(feature);
	}

	return features;
}

pcl::CorrespondencesPtr com_correspondence(pcl::PointCloud<pcl::PFHSignature125>::Ptr source_descriptors, pcl::PointCloud<pcl::PFHSignature125>::Ptr target_descriptors, float dis) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::PFHSignature125> match_search;   //设置配准的方法
	match_search.setInputCloud(target_descriptors);  //输入模板点云的描述子


  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < source_descriptors->size(); ++i)
	{
		std::vector<int> neigh_indices(1);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(1); //申明最近邻平方距离值
		int found_neighs = match_search.nearestKSearch(source_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
		//scene_descriptors->at (i)是给定点云 1是临近点个数 ，neigh_indices临近点的索引  neigh_sqr_dists是与临近点的索引

		if (found_neighs == 1 && neigh_sqr_dists[0] < dis) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(static_cast<int> (i), neigh_indices[0], neigh_sqr_dists[0]);
			model_scene_corrs->push_back(corr);   //把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}

pcl::CorrespondencesPtr com_correspondence2(pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_source, vector<float> dis_source, pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_target, vector<float> dis_target, float dis, float leaf_size) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::PFHSignature125> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子


  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < feature_source->size(); i++)
	{

		float dis_feature = 0.0f;
		for (int j = 0; j < feature_source->points[i].descriptorSize(); j++) {
			dis_feature += pow(feature_source->points[i].histogram[j] - feature_target->points[i].histogram[j], 2.0f);

		}
		dis_feature = sqrt(dis_feature);
		if (dis_feature < dis && abs(dis_source[i] - dis_target[i]) < 10.0f*leaf_size) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(i, i, dis_feature);
			model_scene_corrs->push_back(corr);//把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}

pcl::PointCloud<pcl::PFHSignature125>::Ptr com_features2(pcl::PointCloud<pcl::PointXYZ>::Ptr key, pcl::PointCloud<pcl::PFHSignature125>::Ptr features, vector<float>& dis) {
	pcl::PointCloud<pcl::PFHSignature125>::Ptr new2_features(new pcl::PointCloud<pcl::PFHSignature125>);
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_key(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	kdtree_key->setInputCloud(key);
	for (int i = 0; i < key->size(); i++) {
		std::vector<int> neigh_indices(2);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(2); //申明最近邻平方距离值
		kdtree_key->nearestKSearch(key->at(i), 2, neigh_indices, neigh_sqr_dists);
		pcl::PFHSignature125 feature;
		for (int j = 0; j < feature.descriptorSize(); j++) {
			feature.histogram[j] = features->points[i].histogram[j] + features->points[neigh_indices[1]].histogram[j];
		}
		dis.push_back(sqrt(neigh_sqr_dists[1]));
		new2_features->push_back(feature);

	}
	return new2_features;
}

void show_coor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scenes, pcl::PointCloud<pcl::PointXYZ> keypoints_model, pcl::PointCloud<pcl::PointXYZ> keypoints_scenes, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_model, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_scenes, pcl::CorrespondencesPtr corr) {
	for (int i = 0; i < corr->size(); i++) {
		cout << corr->at(i).index_query << "---" << corr->at(i).index_match << "---" << corr->at(i).distance << endl;

		pcl::visualization::PCLPlotter plotter;
		plotter.addFeatureHistogram<pcl::PFHSignature125>(*features_model, "pfh", corr->at(i).index_query);
		plotter.addFeatureHistogram<pcl::PFHSignature125>(*features_scenes, "pfh", corr->at(i).index_match);
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_model(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_scenes(new pcl::PointCloud<pcl::PointXYZ>());
		keypoints_ptr_model->push_back(keypoints_model.points[corr->at(i).index_query]);
		keypoints_ptr_scenes->push_back(keypoints_scenes.points[corr->at(i).index_match]);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		int v1(0);
		viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v1);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_model(keypoints_ptr_model, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_model, color_key_model, "color_key_model", v1);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_model");

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_model(cloud_model, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_model, color_cloud_model, "cloud_model", v1);

		int v2(0);
		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v2);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_scenes(keypoints_ptr_scenes, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_scenes, color_key_scenes, "color_key_scenes", v2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_scenes");

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_scenes(cloud_scenes, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_scenes, color_cloud_scenes, "cloud_scenes", v2);

		plotter.plot();
		// 等待直到可视化窗口关闭
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			//boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}

}

float com_overlapping(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_target, float leaf_size) {
	int k = 1;
	int count = 0;
	vector<int> id(k);//最近点索引
	vector<float> dist(k);//最近点距离
	for (int i = 0; i < cloud_source->size(); i++) {

		if (kdtree_target->nearestKSearch(cloud_source->points[i], k, id, dist) > 0) {

			if (dist[0] <= leaf_size * 5)//如果搜索到的第一个点距为0，那么这个点为重合点（因为用A搜B中的点，不存在包含自身的情况）,可以自定义一个距离
			{
				count++;
			}

		}

	}
	cout << "重叠率： " << (float)count / (float)cloud_source->size() << endl;
	return (float)count / (float)cloud_source->size();
}

void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, pcl::CorrespondencesPtr corr, float leaf_size) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());


	for (int i = 0; i < corr->size(); i++) {

		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);

	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < cloud_source->size(); i++) {
		new_cloud_source->points[i].y += 500.0f* leaf_size;
	}

	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].y += 500.0f* leaf_size;
	}

	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 255.0, 0.0), "cloud_target");
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0.0, 0.0, 255.0), "cloud_source");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>color_new_key_target(new_key_target, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_target, color_new_key_target, "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_target");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_new_key_source(new_key_source, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_source, color_new_key_source, "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_source");


	for (int i = 0; i < new_key_source->size(); i++)
	{
		pcl::PointXYZ source_point = new_key_source->points[i];
		pcl::PointXYZ target_point = new_key_target->points[i];
		line.addLine(source_point, target_point, 255, 0, 255, to_string(i));
	}
	line.spin();
}

void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, vector<int> corr, float leaf_size) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());



	for (int i = 0; i < corr.size(); i++) {

		new_key_source->push_back(key_source->points[corr[i]]);
		new_key_target->push_back(key_target->points[corr[i]]);

	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < cloud_source->size(); i++) {
		new_cloud_source->points[i].y += 500.0f* leaf_size;
	}

	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].y += 500.0f* leaf_size;
	}

	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 255.0, 0.0), "cloud_target");
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0.0, 0.0, 255.0), "cloud_source");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>color_new_key_target(new_key_target, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_target, color_new_key_target, "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_target");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_new_key_source(new_key_source, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_source, color_new_key_source, "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_source");


	for (int i = 0; i < new_key_source->size(); i++)
	{

		pcl::PointXYZ source_point = new_key_source->points[i];
		pcl::PointXYZ target_point = new_key_target->points[i];
		line.addLine(source_point, target_point, 255, 0, 255, to_string(i));
	}
	line.spin();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr add_gaussian_noise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float m) {
	float leaf_size = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);//Acloud在Bcloud中进行搜索
	//进行1邻域点搜索
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//最近点索引
	std::vector<float> pointNKNSquaredDistance(K);//最近点距离
	//在B点云中计算点与最近邻的平均距离
	double avgdistance = 0;
	for (int i = 0; i < cloud->size(); i++) {
		kdtree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
		leaf_size = leaf_size + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	leaf_size = (float)leaf_size / (float)(cloud->size());
	//添加高斯噪声
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudfiltered(new pcl::PointCloud<pcl::PointXYZ>());
	cloudfiltered->points.resize(cloud->points.size());//将点云的cloud的size赋值给噪声
	cloudfiltered->header = cloud->header;
	cloudfiltered->width = cloud->width;
	cloudfiltered->height = cloud->height;
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(time(0)));
	boost::normal_distribution<> nd(0, m*leaf_size);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> var_nor(rng, nd);
	//添加噪声
	for (size_t point_i = 0; point_i < cloud->points.size(); ++point_i)
	{
		cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].y = cloud->points[point_i].y + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].z = cloud->points[point_i].z + static_cast<float> (var_nor());
	}
	return cloudfiltered;
}

int main() {
	float multiple = 5.0f;
	double start = 0;
	double end = 0;

	string road = "d:/code/PCD/自建配准点云/scene+rt/";
	vector<string> names = { "a1", "a2","a2rt","angel1","angel2","angel2rt","b1","b2","b2rt","c1","c2", "c2rt","d1","d2","d2rt","g1","g2","g2rt","h1","h2","h2rt","hand1","hand2","hand2rt","horse1","horse2","horse2rt" };
	//////////////////////////0//////1/////2//////3////////4//////////5//////6/////7////8/////9////10////11/////12///13////14////15///16////17/////18///19////20/////21/////22///////23///////24///////25/////////26///////
	string name_source = road + names[18] + ".ply";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile(name_source, *cloud_source);

	string name_target = road + names[20] + ".ply";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile(name_target, *cloud_target);

	//*cloud_source = *add_gaussian_noise(cloud_source, 0.5);
	//*cloud_target = *add_gaussian_noise(cloud_target, 0.5);

	//pcl::visualization::PCLVisualizer visu0("before");
	//visu0.setBackgroundColor(255, 255, 255);
	//visu0.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0, 255.0), "cloud_target");
	//visu0.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255, 0), "cloud_source");
	//visu0.spin();


	///////////////源点云/////////////////////////
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_source(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	kdtree_source->setInputCloud(cloud_source);
	float leaf_size = com_leaf_size(cloud_source, kdtree_source);
	cout << "源点云分辨率：" << leaf_size << endl;
	cout << "源点云点数：" << cloud_source->size() << endl;
	///////////////源点云法线/////////////////////////
	start = GetTickCount();
	pcl::PointCloud<pcl::Normal>::Ptr normal_source(new pcl::PointCloud<pcl::Normal>);
	*normal_source = *normal_estimation_OMP(cloud_source, leaf_size);
	end = GetTickCount();
	cout << "源点云法线：" << end - start << "ms" << endl;
	///////////////源点云关键点///////////////////////////////////////
	start = GetTickCount();
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);
	vector<float> ls;
	*key_source = *key_detect(cloud_source, normal_source, kdtree_source, multiple*leaf_size, ls);
	end = GetTickCount();
	cout << "源点云关键点数目：" << key_source->size() << endl;
	cout << "源点云关键点：" << end - start << "ms" << endl;

	///////////////目标点云/////////////////////////
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_target(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	kdtree_target->setInputCloud(cloud_target);
	float leaf_size2 = com_leaf_size(cloud_target, kdtree_target);
	cout << "目标点云分辨率：" << leaf_size2 << endl;
	cout << "目标点云点数：" << cloud_target->size() << endl;
	///////////////目标点云法线/////////////////////////
	start = GetTickCount();
	pcl::PointCloud<pcl::Normal>::Ptr normal_target(new pcl::PointCloud<pcl::Normal>);
	*normal_target = *normal_estimation_OMP(cloud_target, leaf_size);
	end = GetTickCount();
	cout << "目标点云法线：" << end - start << "ms" << endl;
	///////////////目标点云关键点///////////////////////////////////////
	start = GetTickCount();
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target(new pcl::PointCloud<pcl::PointXYZ>);

	vector<float> lt;
	*key_target = *key_detect(cloud_target, normal_target, kdtree_target, multiple*leaf_size, lt);
	end = GetTickCount();
	cout << "目标点云关键点数目：" << key_target->size() << endl;
	cout << "目标点云关键点：" << end - start << "ms" << endl;

	/////////////////最终关键点//////////////////////////////////////////////////
	finall_key(key_source, ls, key_target, lt);
	cout << "源点云关键点数目：" << key_source->size() << endl;
	cout << "目标点云关键点数目：" << key_target->size() << endl;
	//show_key_point(cloud_source, key_source);
	//show_key_point(cloud_target, key_target);
	////////////////源点云特征描述//////////////////////////////////////
	start = GetTickCount();
	pcl::PointCloud<pcl::PFHSignature125>::Ptr features_source(new pcl::PointCloud<pcl::PFHSignature125>);
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf_source(new pcl::PointCloud<pcl::ReferenceFrame>());
	*features_source = *com_features(cloud_source, key_source, normal_source, kdtree_source, *rf_source, leaf_size, multiple);
	end = GetTickCount();
	cout << "源点云特征描述：" << end - start << "ms" << endl;



	////////////////目标点云特征描述//////////////////////////////////////
	start = GetTickCount();
	pcl::PointCloud<pcl::PFHSignature125>::Ptr features_target(new pcl::PointCloud<pcl::PFHSignature125>);
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf_target(new pcl::PointCloud<pcl::ReferenceFrame>());
	*features_target = *com_features(cloud_target, key_target, normal_target, kdtree_target, *rf_target, leaf_size, multiple);
	end = GetTickCount();
	cout << "目标点云特征描述：" << end - start << "ms" << endl;

	//show_key_cloud(cloud_source, key_source, cloud_target, key_target);

	////////////////////初始对应关系/////////////////////////////
	start = GetTickCount();
	pcl::CorrespondencesPtr corr(new pcl::Correspondences());
	float dis = 0.012;
	*corr = *com_correspondence(features_source, features_target, dis);
	end = GetTickCount();
	cout << "初始对应关系数目：" << corr->size() << endl;
	cout << "初始对应关系：" << end - start << "ms" << endl;
	//show_coor(cloud_source, cloud_target, *key_source, *key_target, features_source, features_target, corr);
	show_line(cloud_source, cloud_target, key_source, key_target, corr, leaf_size);
	pcl::registration::CorrespondenceRejectorOneToOne c;
	c.setInputCorrespondences(corr);
	c.getCorrespondences(*corr);



	////////////////////将对应关系存入点云//////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PFHSignature125>::Ptr new_features_source(new pcl::PointCloud<pcl::PFHSignature125>);
	pcl::PointCloud<pcl::PFHSignature125>::Ptr new_features_target(new pcl::PointCloud<pcl::PFHSignature125>);
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr new_rf_source(new pcl::PointCloud<pcl::ReferenceFrame>());
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr new_rf_target(new pcl::PointCloud<pcl::ReferenceFrame>());
	for (int i = 0; i < corr->size(); i++) {

		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);
		new_features_source->push_back(features_source->points[corr->at(i).index_query]);
		new_features_target->push_back(features_target->points[corr->at(i).index_match]);
		new_rf_source->push_back(rf_source->points[corr->at(i).index_query]);
		new_rf_target->push_back(rf_target->points[corr->at(i).index_match]);
	}



	////////////////////去除错误对应关系////////////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::PointCloud<pcl::PFHSignature125>::Ptr new2_features_source(new pcl::PointCloud<pcl::PFHSignature125>);
	vector<float> dis_source;

	pcl::PointCloud<pcl::PFHSignature125>::Ptr new2_features_target(new pcl::PointCloud<pcl::PFHSignature125>);
	vector<float> dis_target;

	*new2_features_source = *com_features2(new_key_source, new_features_source, dis_source);
	*new2_features_target = *com_features2(new_key_target, new_features_target, dis_target);


	pcl::CorrespondencesPtr corr2(new pcl::Correspondences());
	float dis2 = 0.15;
	*corr2 = *com_correspondence2(new2_features_source, dis_source, new2_features_target, dis_target, dis2, leaf_size);
	end = GetTickCount();
	cout << "对应关系数目：" << corr2->size() << endl;
	cout << "对应关系：" << end - start << "ms" << endl;



	pcl::PointCloud<pcl::PointXYZ>::Ptr new2_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new2_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr new2_rf_source(new pcl::PointCloud<pcl::ReferenceFrame>());
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr new2_rf_target(new pcl::PointCloud<pcl::ReferenceFrame>());
	for (int i = 0; i < corr2->size(); i++) {

		new2_key_source->push_back(new_key_source->points[corr2->at(i).index_query]);
		new2_key_target->push_back(new_key_target->points[corr2->at(i).index_match]);
		new2_rf_source->push_back(new_rf_source->points[corr2->at(i).index_query]);
		new2_rf_target->push_back(new_rf_target->points[corr2->at(i).index_match]);
	}
	//show_coor(cloud_source, cloud_target, *new_key_source, *new_key_target, new2_features_source, new2_features_target, corr2);
	show_line(cloud_source, cloud_target, new_key_source, new_key_target, corr2, leaf_size);

	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float> svd;
	Eigen::Matrix4f trans;
	svd.estimateRigidTransformation(*new2_key_source, *new2_key_target, trans);
	cout << trans;
	pcl::transformPointCloud(*cloud_source, *cloud_source, trans);

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud_target(new pcl::PointCloud<pcl::PointXYZ>());
	*sample_cloud_source = *voxel_grid(cloud_source, 2.0*leaf_size);
	*sample_cloud_target = *voxel_grid(cloud_target, 3.0*leaf_size);
	pcl::visualization::PCLVisualizer ranc_view("ranc_view");
	ranc_view.setBackgroundColor(255, 255, 255);
	ranc_view.addPointCloud(sample_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sample_cloud_source, 0, 255, 0), "ranc_scene");
	ranc_view.addPointCloud(sample_cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sample_cloud_target, 0.0, 0.0, 255), "ranc_target");
	ranc_view.spin();

	//pcl::visualization::PCLVisualizer ranc_view("ranc_view");
	//ranc_view.setBackgroundColor(255, 255, 255);
	//ranc_view.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0, 255, 0), "new_cloud_source");
	//ranc_view.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 255), "cloud_target");
	//ranc_view.spin();


	///////////////////////////ranc去除错误对应关系//////////////////////////////////////
	//start = GetTickCount();
	//pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::PFHSignature125> align;
	//align.setInputSource(new_key_source);
	//align.setSourceFeatures(new_features_source);
	//align.setInputTarget(new_key_target);
	//align.setTargetFeatures(new_features_target);
	//align.setMaximumIterations(1000); // Number of RANSAC iterations
	//align.setNumberOfSamples(3); // Number of points to sample for generating/prerejecting a pose
	//align.setCorrespondenceRandomness(3); // Number of nearest features to use
	//align.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
	//align.setMaxCorrespondenceDistance(leaf_size); // Inlier threshold
	//align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesi

	//pcl::PointCloud<pcl::PointXYZ>::Ptr final_new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	//align.align(*final_new_key_source);//计算坐标变换，且存储变换后点云
	//end = GetTickCount();
	//cout << "ranc：" << end - start << "ms" << endl;
	//cout << "ranc分数： " << align.getFitnessScore(leaf_size) << endl;//计算分数是会对源点云进行变换，所以之前用另一个点云存储变化后点云
	//pcl::console::print_info("内点数： %i/%i\n", align.getInliers().size(), new_key_source->size());
	//cout << "/////////////////////////////////////////////////////" << endl;
	//cout << align.getFinalTransformation() << endl;
	//cout << "/////////////////////////////////////////////////////" << endl;
	//pcl::transformPointCloud(*cloud_source, *cloud_source, align.getFinalTransformation());
	//vector<int> corr2=align.getInliers();
	//show_line(cloud_source, cloud_target, final_new_key_source, new_key_target, corr2, leaf_size);
	//pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud_source(new pcl::PointCloud<pcl::PointXYZ>());
	//pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud_target(new pcl::PointCloud<pcl::PointXYZ>());
	//*sample_cloud_source = *voxel_grid(cloud_source, 2.0*leaf_size);
	//*sample_cloud_target = *voxel_grid(cloud_target, 3.0*leaf_size);
	//pcl::visualization::PCLVisualizer ranc_view("ranc_view");
	//ranc_view.setBackgroundColor(255, 255, 255);
	//ranc_view.addPointCloud(sample_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sample_cloud_source, 0, 255, 0), "ranc_scene");
	//ranc_view.addPointCloud(sample_cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sample_cloud_target, 0.0, 0.0, 255), "ranc_target");
	//ranc_view.spin();
	/////////////////////////////////////////////icp/////////////////////////////////////////////////////////////////////
	//start = GetTickCount();
	//pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//icp.setInputSource(cloud_source);
	//icp.setInputTarget(cloud_target);
	//icp.setMaxCorrespondenceDistance(2.0f*leaf_size);
	//icp.setMaximumIterations(30);
	//pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud_source(new pcl::PointCloud<pcl::PointXYZ>());
	//icp.align(*final_cloud_source);
	//end = GetTickCount();
	//cout << "icp：" << end - start << "ms" << endl;
	//std::cout << " icp分数： " << icp.getFitnessScore(leaf_size) << std::endl;
	//cout << "/////////////////////////////////////////////////////" << endl;
	//std::cout << icp.getFinalTransformation() << std::endl;
	//cout << "/////////////////////////////////////////////////////" << endl;
	//pcl::transformPointCloud(*cloud_source, *cloud_source, icp.getFinalTransformation());
	//*sample_cloud_source = *voxel_grid(cloud_source, 2.0*leaf_size);
	//*sample_cloud_target = *voxel_grid(cloud_target, 3.0*leaf_size);
	//pcl::visualization::PCLVisualizer icp_view("icp_view");
	//icp_view.setBackgroundColor(255, 255, 255);
	//icp_view.addPointCloud(sample_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sample_cloud_source, 0, 255, 255), "icp_scene");
	//icp_view.addPointCloud(sample_cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sample_cloud_target, 0.0, 0.0, 255.0), "icp_target");
	//icp_view.spin();

	return 0;
}