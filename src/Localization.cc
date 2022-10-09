/**
* This file is part of OA-SLAM.
*
* Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
* (Inria, LORIA, Université de Lorraine)
* OA-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OA-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Localization.h"

#include <numeric>

#include "p3p.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include "Converter.h"
#include "OptimizerObject.h"


namespace ORB_SLAM2
{

std::pair<Mapping, Mapping> generate_possible_mappings(const std::vector<size_t> objects_category, const std::vector<size_t> detections_category)
{
    // generate objects category <-> id mapping
    std::unordered_map<size_t, std::vector<size_t>> obj_cat_id_mapping;
    for (size_t i = 0; i < objects_category.size(); ++i)
    {
        auto cat = objects_category[i];
        if (obj_cat_id_mapping.find(cat) != obj_cat_id_mapping.end())
        {
            obj_cat_id_mapping[cat].push_back(i);
        }
        else
        {
            obj_cat_id_mapping[cat] = {i};
        }
    }

    // generate detections category <-> id mapping
    std::unordered_map<size_t, std::vector<size_t>> det_cat_id_mapping;
    for (size_t i = 0; i < detections_category.size(); ++i)
    {
        auto cat = detections_category[i];
        if (det_cat_id_mapping.find(cat) != det_cat_id_mapping.end())
        {
            det_cat_id_mapping[cat].push_back(i);
        }
        else
        {
            det_cat_id_mapping[cat] = {i};
        }
    }


    // Generate all possible mapping (detection => possible objects)
    Mapping det_to_obj_mapping(detections_category.size());
    for (size_t i = 0; i < detections_category.size(); ++i)
    {
        det_to_obj_mapping[i] = obj_cat_id_mapping[detections_category[i]];
    }



    // Generate all possible mapping (objects => possible detections)
    Mapping obj_to_det_mapping(objects_category.size());
    for (size_t i = 0; i < objects_category.size(); ++i)
    {
        obj_to_det_mapping[i] = det_cat_id_mapping[objects_category[i]];
    }


    return std::make_pair(det_to_obj_mapping, obj_to_det_mapping);
}


std::tuple<int,
std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>,
std::vector<double>,
std::vector<std::vector<std::pair<size_t, size_t>>>,
std::vector<std::vector<std::pair<size_t, size_t>>>>
solveP3P_ransac(const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                const std::vector<size_t>& ellipsoids_categories,
                const std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& detections,
                const std::vector<size_t>& detections_categories, const Eigen::Matrix3d& K, double THRESHOLD)
{
    if (detections.size() < 3)
    {
        return std::make_tuple(-1, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>(),
                               std::vector<double>(),
                               std::vector<std::vector<std::pair<size_t, size_t>>>(),
                               std::vector<std::vector<std::pair<size_t, size_t>>>());
    }
    Eigen::Matrix3d K_inv = K.inverse();

    // std::vector<Eigen::Vector2d> detections_centers(detections.size());
    std::vector<Eigen::Vector3d> detections_centers_rays(detections.size());
    for (size_t i = 0; i < detections.size(); ++i)
    {
        auto center = bbox_center(detections[i]);
        // detections_centers[i] = ;
        detections_centers_rays[i] = K_inv * center.homogeneous();
    }

    std::pair<Mapping, Mapping> ret = generate_possible_mappings(ellipsoids_categories, detections_categories);
    const Mapping &det_to_obj = ret.first;
    const Mapping &obj_to_det = ret.second;

    std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> all_poses;
    std::vector<double> all_costs;
    int best_index = -1;
    double best_all_costs = std::numeric_limits<double>::infinity();
    std::vector<std::vector<std::pair<size_t, size_t>>> used_pairs;
    std::vector<std::vector<std::pair<size_t, size_t>>> used_inliers;

    for (size_t det_0 = 0; det_0 < det_to_obj.size(); ++det_0)
    {
        for (size_t det_1 = det_0 + 1; det_1 < det_to_obj.size(); ++det_1)
        {
            for (size_t det_2 = det_1 + 1; det_2 < det_to_obj.size(); ++det_2)
            {
                for (size_t obj_0 : det_to_obj[det_0])
                {
                    for (size_t obj_1: det_to_obj[det_1])
                    {
                        if (obj_0 == obj_1) 
                            continue;

                        for (size_t obj_2: det_to_obj[det_2])
                        {
                            if (obj_1 == obj_2 || obj_0 == obj_2)
                                continue;
                            
                            Eigen::Matrix3d X;
                            X.col(0) = ellipsoids[obj_0].GetCenter();
                            X.col(1) = ellipsoids[obj_1].GetCenter();
                            X.col(2) = ellipsoids[obj_2].GetCenter();
                            Eigen::Matrix3d x;
                            x.col(0) = detections_centers_rays[det_0];
                            x.col(1) = detections_centers_rays[det_1];
                            x.col(2) = detections_centers_rays[det_2];
                            x.col(0).normalize();
                            x.col(1).normalize();
                            x.col(2).normalize();

                            Eigen::Matrix<Eigen::Matrix<double, 3, 4>, 4, 1> solutions;
                            monocular_pose_estimator::P3P::computePoses(x, X, solutions);

                            double best_cost = std::numeric_limits<double>::infinity();
                            std::vector<std::pair<size_t, size_t>> best_inliers;
                            Matrix34d best_pose;
                            for (size_t idx = 0; idx < 4; ++idx)
                            {
                                const Eigen::Matrix<double, 3, 4>& p = solutions(idx, 0);
                                Eigen::Matrix3d o = p.block<3, 3>(0, 0);
                                Eigen::Vector3d pos_est = p.col(3);

                                // Check that the camera is in front of the scene
                                if ((ellipsoids[obj_0].GetCenter() - pos_est).dot(o.col(2)) < 0)
                                    continue;

                                // ----- Evaluate coherence -----
                                // project all ellipsoids
                                Matrix34d pose, Rt;
                                pose << o, pos_est;
                                Rt << o.transpose(), -o.transpose() * pos_est;

                                Matrix34d P = K * Rt;
                                // std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> proj_ellipsoids(ellipsoids.size());
                                std::vector<BBox2, Eigen::aligned_allocator<BBox2>> proj_ellipsoids_bboxes(ellipsoids.size());
                                for (size_t k = 0; k < ellipsoids.size(); ++k) {
                                    // proj_ellipsoids[k] = ellipsoids[k].project(P);
                                    proj_ellipsoids_bboxes[k] = ellipsoids[k].project(P).ComputeBbox();
                                }

                                double cost_total = 0.0;
                                std::vector<std::pair<size_t, size_t>> inliers;
                                std::vector<bool> used(detections.size(), false);
                                std::vector<double> costs(detections.size(), 1.0);
                                for (size_t obj_id = 0; obj_id < ellipsoids.size(); ++obj_id)
                                {
                                    double max_iou = 0.0;
                                    int best_det_id = 0;
                                    // For the two ellipsoids used to compute the pose, only used prediction can be matched
                                    if (obj_id == obj_0)
                                    {
                                        // double iou = compute_iou_toms(detections_ellipses_decomposed[det_0][pred_0], proj_ellipsoids[obj_id]);
                                        double iou = bboxes_iou(detections[det_0], proj_ellipsoids_bboxes[obj_id]);
                                        max_iou = iou;
                                        best_det_id = det_0;
                                        used[det_0] = true;
                                    }
                                    else if (obj_id == obj_1)
                                    {
                                        // double iou = compute_iou_toms(detections_ellipses_decomposed[det_1][pred_1], proj_ellipsoids[obj_id]);
                                        double iou = bboxes_iou(detections[det_1], proj_ellipsoids_bboxes[obj_id]);
                                        max_iou = iou;
                                        best_det_id = det_1;
                                        used[det_1] = true;
                                    }
                                    else if (obj_id == obj_2)
                                    {
                                        // double iou = compute_iou_toms(detections_ellipses_decomposed[det_2][pred_2], proj_ellipsoids[obj_id]);
                                        double iou = bboxes_iou(detections[det_2], proj_ellipsoids_bboxes[obj_id]);
                                        max_iou = iou;
                                        best_det_id = det_2;
                                        used[det_2] = true;
                                    }
                                    else
                                    {
                                        for (int det_id : obj_to_det[obj_id])
                                        {
                                            if (used[det_id])
                                                continue;
                                            // double iou = compute_iou_toms(detections_ellipses_decomposed[det_id][pred_id], proj_ellipsoids[obj_id]);
                                            double iou = bboxes_iou(detections[det_id], proj_ellipsoids_bboxes[obj_id]);
                                            if (iou > max_iou)
                                            {
                                                max_iou = iou;
                                                best_det_id = det_id;
                                            }
                                        }
                                    }
                                    if (max_iou >= THRESHOLD || obj_id == obj_0 || obj_id == obj_1 || obj_id == obj_2)
                                    {
                                        inliers.push_back(std::make_pair(best_det_id, obj_id));
                                        // avoid the same detection to be used several times
                                        used[best_det_id] = true;
                                        costs[best_det_id] = 1.0 - max_iou;
                                    }
                                }
                                cost_total = std::accumulate(costs.begin(), costs.end(), 0.0);

                                if (cost_total < best_cost)
                                {
                                    best_cost = cost_total;
                                    best_inliers = inliers;
                                    best_pose = pose;
                                }
                            }


                            if (best_cost < detections.size())
                            {
                                if (best_cost < best_all_costs)
                                {
                                    best_all_costs = best_cost;
                                    best_index = all_costs.size();
                                }
                                all_costs.push_back(best_cost);
                                all_poses.push_back(best_pose);
                                used_pairs.push_back({std::make_pair(det_0, obj_0),
                                                      std::make_pair(det_1, obj_1),
                                                      std::make_pair(det_2, obj_2)});
                                used_inliers.push_back(std::move(best_inliers));
                            }
                        }
                    }
                }
            }
        }
    }

    return std::make_tuple(best_index, std::move(all_poses), std::move(all_costs), std::move(used_pairs), std::move(used_inliers));
}




cv::Mat OptimizePoseFromObjects(const std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>& ellipses,
                                const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                                cv::Mat Rt, const Eigen::Matrix3d& K)
{

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::Solver *solver_ptr = nullptr;

    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    solver_ptr = new g2o::BlockSolver_6_3(linearSolver);


    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);



    g2o::VertexSE3Expmap * vertex_cam = new g2o::VertexSE3Expmap();
    vertex_cam->setEstimate(Converter::toSE3Quat(Rt));
    vertex_cam->setId(0);
    optimizer.addVertex(vertex_cam);

    const int N = ellipsoids.size();
    for (int i = 0; i < N; ++i) {
        EdgeSE3ProjectEllipsoidOnlyPose *e = new EdgeSE3ProjectEllipsoidOnlyPose(ellipses[i], ellipsoids[i], K);
        e->setVertex(0, vertex_cam);
        Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
        e->setInformation(information_matrix);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(50);

    g2o::SE3Quat SE3quat = vertex_cam->estimate();
    return Converter::toCvMat(SE3quat);
}

}
