#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/functional/hash.hpp>
#include "RobotNav2dActions.hpp"
#include <planners/WastarPlanner.hpp>
#include <planners/PwastarPlanner.hpp>
#include <planners/ArastarPlanner.hpp>
#include <planners/PasePlanner.hpp>
#include <planners/EpasePlanner.hpp>
#include <planners/GepasePlanner.hpp>
#include <planners/AgepasePlanner.hpp>
#include <planners/QpasePlanner.hpp>
#include <planners/MplpPlanner.hpp>
#include <planners/BatchPlanner.hpp>

using namespace std;
using namespace ps;

vector<double> goal;

double to_degrees(double rads)
{
    return rads * 180.0 / M_PI;
}

double roundOff(double value, unsigned char prec)
{
    double pow_10 = pow(10.0, (double)prec);
    return round(value * pow_10) / pow_10;
}

vector<vector<double>> loadCostFactorMap(const string& fname, int width, int height)
{
    ifstream infile(fname);
    vector<vector<double>> cost_map(width, vector<double>(height));
    
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            infile >> cost_map[x][y];
        }
    }   

    return cost_map;
}

vector<vector<int>> loadMap(const char *fname, cv::Mat& img, int &width, int &height, int scale=1)
{
    vector<vector<int>> map;
    FILE *f;
    f = fopen(fname, "r");
    
    if (f)
    {
        if (fscanf(f, "type octile\nheight %d\nwidth %d\nmap\n", &height, &width))
        {
            map.resize(width, vector<int>(height));

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    char c;
                    do {
                        int count = fscanf(f, "%c", &c);
                    } while (isspace(c));

                    map[x][y] = (c == '.' || c == 'G' || c == 'S' || c == 'T') ? 0 : 100;
                }
            }            
        }

        fclose(f);
    }

    vector<vector<int>> scaled_map;
    int scaled_height = scale*height;
    int scaled_width = scale*width;
    scaled_map.resize(scaled_width, vector<int>(scaled_height));

    for (int y = 0; y < scaled_height; y++)
    {
        for (int x = 0; x < scaled_width; x++)
        {
            scaled_map[x][y] = map[x/scale][y/scale];
        }
    }

    img = cv::Mat(scaled_height, scaled_width, CV_8UC3);

    for (int y = 0; y < scaled_height; y++)
    {
        for (int x = 0; x < scaled_width; x++)
        {
            img.at<cv::Vec3b>(y,x) = (scaled_map[x][y] > 0) ? cv::Vec3b(0,0,0) : cv::Vec3b(255,255,255);
        }
    }

    height = scaled_height;
    width = scaled_width;
    return scaled_map;

}
double computeHeuristic(const StateVarsType& state_vars, double dist_thresh)
{
    double dist_to_goal_region = pow(pow((state_vars[0] - goal[0]), 2) + pow((state_vars[1] - goal[1]), 2), 0.5);

    if (dist_to_goal_region < dist_thresh)
        dist_to_goal_region = 0;
    return dist_to_goal_region;
}

double computeHeuristicStateToState(const StateVarsType& state_vars_1, const StateVarsType& state_vars_2)
{
    double dist = pow(pow((state_vars_1[0] - state_vars_2[0]), 2) + pow((state_vars_1[1] - state_vars_2[1]), 2), 0.5);
    if (dist < 0)
        dist = 0;
    return dist;
}

void computeDijkstraHeuristic(vector<vector<double>>& heuristic_table, vector<shared_ptr<Action>>& action_ptrs)
{
    // Dijkstra heuristic
    // Resize the heuristic table
    vector<double> map_size = action_ptrs[0]->GetDomainKnowledge();
    double hueristic_noise_factor = action_ptrs[0]->GetParams()["heuristic_noise_factor"];
    vector<vector<double>> heuristic_factor_map = action_ptrs[0]->GetHeuristicFactorMap();
    heuristic_table.resize(map_size[0], vector<double>(map_size[1], DINF));
    // Data Structures
    priority_queue<pair<double, pair<int, int>>, vector<pair<double, pair<int, int>>>, greater<pair<double, pair<int, int>>>> pq;
    vector<vector<bool>> visited(map_size[0], vector<bool>(map_size[1], false));
    // Initialize the priority queue
    pq.push(make_pair(0, make_pair(goal[0], goal[1])));
    
    // Max cost
    double max_cost = 0;

    while (!pq.empty())
    {
        auto top = pq.top();
        pq.pop();
        int x = top.second.first;
        int y = top.second.second;
        double cost = top.first;

        if (visited[x][y])
            continue;

        visited[x][y] = true;
        double noise = 0;
        if (!heuristic_factor_map.empty())
            noise = (heuristic_factor_map[x][y] > 50) ? hueristic_noise_factor*(heuristic_factor_map[x][y] - 50) : 0;
        heuristic_table[x][y] = cost + noise;
        max_cost = cost;
        
        // cout << "x: " << x << " y: " << y << " cost: " << cost << endl;

        for (auto& action_ptr : action_ptrs)
        {
            auto succ = action_ptr->GetSuccessor({static_cast<double>(x), static_cast<double>(y), 0}, 0);
            if (succ.success_)
            {
                auto succ_state = succ.successor_state_vars_costs_[0].first;
                int succ_x = static_cast<int>(succ_state[0]);
                int succ_y = static_cast<int>(succ_state[1]);
                double succ_cost = succ.successor_state_vars_costs_[0].second;
                if (!visited[succ_x][succ_y])
                {
                    pq.push(make_pair(cost + succ_cost, make_pair(succ_x, succ_y)));
                }
            }
        }
    }
    // cv::Mat img(map_size[1], map_size[0], CV_8UC3);
    // for (int y = 0; y < map_size[1]; y++)
    // {
    //     for (int x = 0; x < map_size[0]; x++)
    //     {
    //         img.at<cv::Vec3b>(y,x) = (heuristic_table[x][y] < DINF) ? cv::Vec3b(255,255,0)*heuristic_table[x][y]/max_cost : cv::Vec3b(0,0,0) ;
    //     }
    // }
    //
    // cv::imshow("Dijkstra Heuristic", img);
    // cv::waitKey(0);
}


bool isGoalState(const StateVarsType& state_vars, double dist_thresh)
{
    return (computeHeuristic(state_vars, dist_thresh) <= 0);
}

vector<StateVarsType> getExplicitGraph(const StateVarsType& state_vars, vector<shared_ptr<Action>>& action_ptrs)
{
    return action_ptrs[0]->GetExplicitGraph(state_vars);
}

size_t StateKeyGenerator(const StateVarsType& state_vars)
{ 
    int x = round(state_vars[0]); 
    int y = round(state_vars[1]); 
    size_t seed = 0;
    boost::hash_combine(seed, x);
    boost::hash_combine(seed, y);
    return seed;
}

size_t EdgeKeyGenerator(const EdgePtrType& edge_ptr)
{
    int controller_id;
    auto action_ptr = edge_ptr->action_ptr_;

    if (action_ptr == NULL)
        controller_id = 16;
    else if (action_ptr->GetType() ==  "MoveUp")
        controller_id = 0;
    else if (action_ptr->GetType() ==  "MoveUpLong")
        controller_id = 1;
    else if (action_ptr->GetType() ==  "MoveUpRight")
        controller_id = 2;
    else if (action_ptr->GetType() ==  "MoveUpRightLong")
        controller_id = 3;
    else if (action_ptr->GetType() ==  "MoveRight")
        controller_id = 4;
    else if (action_ptr->GetType() ==  "MoveRightLong")
        controller_id = 5;
    else if (action_ptr->GetType() ==  "MoveRightDown")
        controller_id = 6;
    else if (action_ptr->GetType() ==  "MoveRightDownLong")
        controller_id = 7;
    else if (action_ptr->GetType() ==  "MoveDown")
        controller_id = 8;
    else if (action_ptr->GetType() ==  "MoveDownLong")
        controller_id = 9;
    else if (action_ptr->GetType() ==  "MoveDownLeft")
        controller_id = 10;
    else if (action_ptr->GetType() ==  "MoveDownLeftLong")
        controller_id = 11;
    else if (action_ptr->GetType() ==  "MoveLeft")
        controller_id = 12;
    else if (action_ptr->GetType() ==  "MoveLeftLong")
        controller_id = 13;
    else if (action_ptr->GetType() ==  "MoveLeftUp")
        controller_id = 14;
    else if (action_ptr->GetType() ==  "MoveLeftUpLong")
        controller_id = 15;
    else
        throw runtime_error("Controller type not recognized in getEdgeKey!");

    size_t seed = 0;
    boost::hash_combine(seed, edge_ptr->parent_state_ptr_->GetStateID());
    boost::hash_combine(seed, controller_id);


    return seed;
}

void constructActions(vector<shared_ptr<Action>>& action_ptrs, ParamsType& action_params, vector<vector<int>>& map, vector<vector<double>>& cost_factor_map, vector<vector<double>>& heuristic_factor_map)
{
    // Define action parameters
    // action_params["length"] = 25;
    // action_params["footprint_size"] = 16;
    action_params["length"] = 2;
    action_params["footprint_size"] = 1;
    action_params["cache_footprint"] = 1;
    action_params["inflate_evaluation"] = 0;
    action_params["inflate_eval_loops"] = 2500;

    ParamsType expensive_action_params = action_params;
    expensive_action_params["cache_footprint"] = 1;
    
    auto move_up_controller_ptr = make_shared<MoveUpAction>("MoveUp", action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_up_controller_ptr);

    auto move_up_right_controller_ptr = make_shared<MoveUpRightAction>("MoveUpRight", expensive_action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_up_right_controller_ptr);

    auto move_right_controller_ptr = make_shared<MoveRightAction>("MoveRight", action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_right_controller_ptr);

    auto move_right_down_controller_ptr = make_shared<MoveRightDownAction>("MoveRightDown", expensive_action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_right_down_controller_ptr);

    auto move_down_controller_ptr = make_shared<MoveDownAction>("MoveDown", action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_down_controller_ptr);

    auto move_down_left_controller_ptr = make_shared<MoveDownLeftAction>("MoveDownLeft", expensive_action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_down_left_controller_ptr);

    auto move_left_controller_ptr = make_shared<MoveLeftAction>("MoveLeft", action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_left_controller_ptr);

    auto move_left_up_controller_ptr = make_shared<MoveLeftUpAction>("MoveLeftUp", expensive_action_params, map, cost_factor_map, heuristic_factor_map);
    action_ptrs.emplace_back(move_left_up_controller_ptr);

}

void constructPlanner(string planner_name, shared_ptr<Planner>& planner_ptr, vector<shared_ptr<Action>>& action_ptrs, ParamsType& planner_params, ParamsType& action_params)
{
    if (planner_name == "wastar")
        planner_ptr = make_shared<WastarPlanner>(planner_params);
    else if (planner_name == "pwastar")
        planner_ptr = make_shared<PwastarPlanner>(planner_params);
    else if (planner_name == "arastar")
        planner_ptr = make_shared<ArastarPlanner>(planner_params);
    else if (planner_name == "pase")
        planner_ptr = make_shared<PasePlanner>(planner_params);
    else if (planner_name == "epase")
        planner_ptr = make_shared<EpasePlanner>(planner_params); 
    else if (planner_name == "gepase")
        planner_ptr = make_shared<GepasePlanner>(planner_params); 
    else if (planner_name == "agepase")
        planner_ptr = make_shared<AgepasePlanner>(planner_params);
    else if (planner_name == "qpase")
        planner_ptr = make_shared<QpasePlanner>(planner_params);
    else if (planner_name == "mplp")
        planner_ptr = make_shared<MplpPlanner>(planner_params); 
    else if (planner_name == "bplp")
        planner_ptr = make_shared<BatchPlanner>(planner_params);
    else
        throw runtime_error("Planner type not identified!");      

    planner_ptr->SetActions(action_ptrs);
    planner_ptr->SetStateMapKeyGenerator(bind(StateKeyGenerator, placeholders::_1));
    planner_ptr->SetEdgeKeyGenerator(bind(EdgeKeyGenerator, placeholders::_1));
    planner_ptr->SetHeuristicGenerator(bind(computeHeuristic, placeholders::_1, action_params["length"]));
    planner_ptr->SetStateToStateHeuristicGenerator(bind(computeHeuristicStateToState, placeholders::_1, placeholders::_2));
    planner_ptr->SetGoalChecker(bind(isGoalState, placeholders::_1, action_params["length"]));
    planner_ptr->SetExplicitGraph(bind(getExplicitGraph, placeholders::_1, action_ptrs));
    planner_ptr->SetDijkstraHeuristicGenerator(bind(computeDijkstraHeuristic, placeholders::_1, action_ptrs));
}

void loadStartsGoalsFromFile(vector<vector<double>>& starts, vector<vector<double>>& goals, int scale, int num_runs, const string& path)
{
    ifstream starts_fin(path + "nav2d_starts.txt");
    ifstream goals_fin(path + "nav2d_goals.txt");
   
    for (int j = 0; j < num_runs; ++j)
    {
        vector<double> start, goal;
        double val_start, val_goal;
        for (int i = 0; i < 2; ++i)
        {
            starts_fin >> val_start;
            goals_fin >> val_goal;                
            start.push_back(scale*val_start);
            goal.push_back(scale*val_goal);
        }
        start[2] = to_degrees(start[2]);
        goal[2] = to_degrees(goal[2]);
        starts.emplace_back(start);
        goals.emplace_back(goal);

        double cost, length;
        starts_fin >> cost;            
        starts_fin >> length;            
    }
}

/// @brief Log the statistics of a planner to a file.
void logStats(vector<double>& cost_vec, vector<int>& path_length_vec, vector<int>& num_edges_vec, int num_threads, double w, double noise_factor, string planner_name) {
    // save the logs to a file in current directory
    std::string log_file = "exp/euclidean/" + planner_name + "_"  + std::to_string(num_threads) + "_"  + std::to_string(w) + "_"  + std::to_string(noise_factor) + ".csv";
    // save logs object to a file
    std::ofstream file(log_file);
    // header line
    file << "Problem,Time,Cost,Num_expanded,Num_generated," << noise_factor << std::endl;
    for (int i = 0; i < cost_vec.size(); i++) {
        file << i << "," << cost_vec[i] << "," << path_length_vec[i] << "," << num_edges_vec[i] << std::endl;
    }
    file.close();
}

int main(int argc, char* argv[])
{
    int num_threads;
    int batch_size = 2000;
    double time_budget = 0;
    bool apply_cost_factor_map = false;
    bool apply_heuristic_noise = true;
    double heuristic_noise_factor = 0;
    double heuristic_weight = 50;
    double heuristic_reduction = 0.5;
    bool visualize_batch = false;
    bool dijkstra_heuristic = true;

    if (!strcmp(argv[1], "wastar"))
    {
        if (argc == 2) 
            num_threads = 1;
        else if (argc == 3)
        {
            num_threads = 1;
            heuristic_weight = atof(argv[2]);
        }
        else if (argc == 4)
        {
            num_threads = 1;
            heuristic_weight = atof(argv[2]);
            dijkstra_heuristic = atoi(argv[3]);
        }
        else
            throw runtime_error("Format: run_robot_nav_2d wastar");
    }
    else if (!strcmp(argv[1], "mplp"))
    {
        if (argc == 3)
        {
            if (atoi(argv[2]) < 4) throw runtime_error("mplp requires a minimum of 4 threads");
            num_threads = atoi(argv[2]);
        }
        if (argc == 4)
        {
            if (atoi(argv[2]) < 4) throw runtime_error("mplp requires a minimum of 4 threads");
            num_threads = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
        }
        else
            throw runtime_error("Format: run_robot_nav_2d [planner_name] [num_threads] [heuristic_weight]");
    }
    else if (!strcmp(argv[1], "bplp"))
    {
        num_threads = 3;
        visualize_batch = true;
        // visualize_batch = false;
        if (argc == 3)
        {
            // if (atoi(argv[2]) < 4) throw runtime_error("bplp requires a minimum of 4 threads");
            batch_size = atoi(argv[2]);
        }
        if (argc == 4)
        {
            // if (atoi(argv[2]) < 4) throw runtime_error("bplp requires a minimum of 4 threads");
            batch_size = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
        }
        else
            throw runtime_error("Format: run_robot_nav_2d [planner_name] [num_threads] [heuristic_weight]");
    }
    else if (!strcmp(argv[1], "arastar"))
    {
        if (argc == 3)
        {
            num_threads = 1;
            time_budget = atof(argv[2]);
        }
        else if (argc == 5)
        {
            num_threads = 1;
            time_budget = atof(argv[2]);
            heuristic_weight = atof(argv[3]);
            heuristic_reduction = atof(argv[4]);
        }
        else
            throw runtime_error("Format: run_robot_nav_2d arastar [time_budget] [heuristic_weight] [heuristic_reduction]");
    }
    else if (!strcmp(argv[1], "agepase"))
    {
        if (argc == 4)
        {
            num_threads = atoi(argv[2]);
            time_budget = atof(argv[3]);
        }
        else if (argc == 6)
        {
            num_threads = atoi(argv[2]);
            time_budget = atof(argv[3]);
            heuristic_weight = atof(argv[4]);
            heuristic_reduction = atof(argv[5]);
        }
        else
            throw runtime_error("Format: run_robot_nav_2d agepase [num_threads] [time_budget] [heuristic_weight] [heuristic_reduction]");
    }
    else if (!strcmp(argv[1], "epase"))
    {
        if (argc == 3)
        {
            num_threads = atoi(argv[2]);
        }
        else if (argc == 4)
        {
            num_threads = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
        }
        else if (argc == 5)
        {
            num_threads = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
            dijkstra_heuristic= atoi(argv[4]);
        }
        else if (argc == 6)
        {
            num_threads = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
            dijkstra_heuristic= atoi(argv[4]);
            heuristic_noise_factor = atof(argv[5]);
        }
        else
            throw runtime_error("Format: run_robot_nav_2d epase [num_threads] [heuristic_weight]");
    }
    else if (!strcmp(argv[1], "qpase"))
    {
        if (argc == 3)
        {
            num_threads = atoi(argv[2]);
        }
        else if (argc == 4)
        {
            num_threads = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
        }
        else if (argc == 5)
        {
            num_threads = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
            dijkstra_heuristic= atoi(argv[4]);
        }
        else if (argc == 6)
        {
            num_threads = atoi(argv[2]);
            heuristic_weight = atof(argv[3]);
            dijkstra_heuristic= atoi(argv[4]);
            heuristic_noise_factor = atof(argv[5]);
        }
        else
            throw runtime_error("Format: run_robot_nav_2d qpase [num_threads] [heuristic_weight]");
    }
    else
    {
        if (argc != 3) throw runtime_error("Format: run_robot_nav_2d [planner_name] [num_threads]");
        num_threads = atoi(argv[2]);
    }
    

    // Experiment parameters
    int num_runs = 50;
    // vector<int> scale_vec = {2, 2, 2, 4, 2};
    vector<int> scale_vec = {1, 1, 1, 1, 1};
    // vector<int> scale_vec = {5, 5, 5, 10, 5};
    bool visualize_plan = true;
    // bool visualize_plan = false;
    bool load_starts_goals_from_file = true;

    // Define planner parameters
    ParamsType planner_params;
    string planner_name = argv[1];
    planner_params["num_threads"] = num_threads;
    planner_params["heuristic_weight"] = heuristic_weight;
    planner_params["heuristic_reduction"] = heuristic_reduction;
    planner_params["batch_size"] = batch_size;
    planner_params["visualize_batch"] = visualize_batch;
    planner_params["dijkstra_heuristic"] = dijkstra_heuristic;
    if (time_budget)
    {
        planner_params["timeout"] = time_budget;
    }
    else
    {
        planner_params["timeout"] = 20;
    }
    
    // Read map
    int width, height;
    cv::Mat img;
    
    vector<vector<vector<int>>> map_vec;
    vector<vector<vector<double>>> cost_factor_map_vec;
    vector<vector<vector<double>>> heuristic_factor_map_vec; // Use the same as cost factor map for now
    
    vector<cv::Mat> img_vec;

    map_vec.emplace_back(loadMap("../examples/robot_nav_2d/resources/hrt201n/hrt201n.map", img, width, height, scale_vec[0]));
    cost_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/hrt201n/hrt201n_cost_factor.map", width, height));
    heuristic_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/hrt201n/hrt201n_cost_factor.map", width, height));
    img_vec.emplace_back(img.clone());

    map_vec.emplace_back(loadMap("../examples/robot_nav_2d/resources/den501d/den501d.map", img, width, height, scale_vec[1]));
    cost_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/den501d/den501d_cost_factor.map", width, height));
    heuristic_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/den501d/den501d_cost_factor.map", width, height));
    img_vec.emplace_back(img.clone());

    map_vec.emplace_back(loadMap("../examples/robot_nav_2d/resources/den520d/den520d.map", img, width, height, scale_vec[2]));
    cost_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/den520d/den520d_cost_factor.map", width, height));
    heuristic_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/den520d/den520d_cost_factor.map", width, height));
    img_vec.emplace_back(img.clone());

    map_vec.emplace_back(loadMap("../examples/robot_nav_2d/resources/ht_chantry/ht_chantry.map", img, width, height, scale_vec[3]));
    cost_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/ht_chantry/ht_chantry_cost_factor.map", width, height));
    heuristic_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/ht_chantry/ht_chantry_cost_factor.map", width, height));
    img_vec.emplace_back(img.clone());

    map_vec.emplace_back(loadMap("../examples/robot_nav_2d/resources/brc203d/brc203d.map", img, width, height, scale_vec[4]));
    cost_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/brc203d/brc203d_cost_factor.map", width, height));
    heuristic_factor_map_vec.emplace_back(loadCostFactorMap("../examples/robot_nav_2d/resources/brc203d/brc203d_cost_factor.map", width, height));
    img_vec.emplace_back(img.clone());


    vector<string> starts_goals_path = {"../examples/robot_nav_2d/resources/hrt201n/", 
    "../examples/robot_nav_2d/resources/den501d/", 
    "../examples/robot_nav_2d/resources/den520d/",
    "../examples/robot_nav_2d/resources/ht_chantry/",
    "../examples/robot_nav_2d/resources/brc203d/",
    };

    vector<double> all_maps_time_vec, all_maps_cost_vec;
    vector<int> all_maps_path_length_vec, all_maps_num_states_vec, all_maps_num_edges_vec;
    unordered_map<string, vector<double>> all_action_eval_times;

    // for (int m_idx = 0; m_idx < map_vec.size(); ++m_idx)
    int m_idx = 1;
    if (1)
    {
        auto map = map_vec[m_idx];
        auto img = img_vec[m_idx];
        auto cost_factor_map = apply_cost_factor_map ? cost_factor_map_vec[m_idx] : vector<vector<double>>();
        auto heuristic_factor_map = apply_heuristic_noise ? heuristic_factor_map_vec[m_idx] : vector<vector<double>>();
        auto scale = scale_vec[m_idx];

        // Construct actions
        ParamsType action_params;
        action_params["heuristic_noise_factor"] = heuristic_noise_factor;
        vector<shared_ptr<Action>> action_ptrs;
        constructActions(action_ptrs, action_params, map, cost_factor_map, heuristic_factor_map);

        // Construct planner
        shared_ptr<Planner> planner_ptr;
        constructPlanner(planner_name, planner_ptr, action_ptrs, planner_params, action_params);

        // Read starts and goals from text file
        vector<vector<double>> starts, goals;

        if (load_starts_goals_from_file)
            loadStartsGoalsFromFile(starts, goals, scale, num_runs, starts_goals_path[m_idx]);
        else
        {
            starts = vector<vector<double>> (num_runs, {scale*10.0, scale*61.0});
            goals = vector<vector<double>> (num_runs, {scale*200.0, scale*170.0});
        }

        // Run experiments
        int start_goal_idx = 0;
        vector<double> time_vec, cost_vec;
        vector<int> path_length_vec, num_states_vec, num_edges_vec, threads_used_vec;
        vector<int> jobs_per_thread(planner_params["num_threads"], 0);
        unordered_map<string, vector<double>> action_eval_times;

        cout << "Map size: (" << map.size() << ", " << map[0].size() << ") | " 
        << " | Planner: " << planner_name   
        << " | Heuristic weight: " << planner_params["heuristic_weight"]   
        << " | Number of threads: " << planner_params["num_threads"]   
        << " | Number of runs: " << num_runs
        << endl;
        cout <<  "---------------------------------------------------" << endl;

        if (visualize_plan) cv::namedWindow("Plan", cv::WINDOW_AUTOSIZE );// Create a window for display.
        
        int num_success = 0;
        for (int exp_idx = 0; exp_idx < num_runs; ++exp_idx)
        // int exp_idx = 8;
        // if (1)
        {
            cout << "Experiment: " << exp_idx;
            start_goal_idx = exp_idx;
            if (start_goal_idx >= starts.size()) 
                start_goal_idx = 0;

            // Set start state
            planner_ptr->SetStartState(starts[start_goal_idx]);
            
            // Set goal conditions
            goal.clear();
            goal.emplace_back(goals[start_goal_idx][0]);
            goal.emplace_back(goals[start_goal_idx][1]);

            double t=0, cost=0;
            int num_edges=0;

            bool plan_found = planner_ptr->Plan();
            
            if (plan_found)
            {
                auto planner_stats = planner_ptr->GetStats();
                
                time_vec.emplace_back(planner_stats.total_time);
                all_maps_time_vec.emplace_back(planner_stats.total_time);
                cost_vec.emplace_back(planner_stats.path_cost);
                all_maps_cost_vec.emplace_back(planner_stats.path_cost);
                path_length_vec.emplace_back(planner_stats.path_length);
                all_maps_path_length_vec.emplace_back(planner_stats.path_length);
                num_states_vec.emplace_back(planner_stats.num_state_expansions);
                all_maps_num_states_vec.emplace_back(planner_stats.num_state_expansions);
                num_edges_vec.emplace_back(planner_stats.num_evaluated_edges);
                all_maps_num_edges_vec.emplace_back(planner_stats.num_evaluated_edges);

                for (auto& [action, times] : planner_stats.action_eval_times)
                { 
                    action_eval_times[action].insert(action_eval_times[action].end(), times.begin(), times.end());
                    all_action_eval_times[action].insert(all_action_eval_times[action].end(), times.begin(), times.end());
                }

                threads_used_vec.emplace_back(planner_stats.num_threads_spawned);
                cout << " | Time (s): " << planner_stats.total_time 
                << " | Cost: " << planner_stats.path_cost 
                << " | Length: " << planner_stats.path_length
                << " | State expansions: " << planner_stats.num_state_expansions
                << " | Edges evaluated: " << planner_stats.num_evaluated_edges
                << " | Threads used: " << planner_stats.num_threads_spawned << "/" << planner_params["num_threads"]
                << " | Lock time: " <<  planner_stats.lock_time
                << " | Expand time: " << planner_stats.cumulative_expansions_time
                << " | Threads: " << planner_stats.num_threads_spawned << "/" << planner_params["num_threads"] << endl;
               
                // cout << endl << "------------- Jobs per thread -------------" << endl;
                // for (int tidx = 0; tidx < planner_params["num_threads"]; ++tidx)
                    // cout << "thread: " << tidx << " jobs: " << planner_stats.num_jobs_per_thread[tidx] << endl;
                for (int tidx = 0; tidx < planner_params["num_threads"]; ++tidx)
                    jobs_per_thread[tidx] += planner_stats.num_jobs_per_thread[tidx];        
                
                num_success++;
            }
            else
                cout << " | Plan not found!" << endl;

            ++start_goal_idx;   

            if (visualize_plan)
            {
                cv::Mat img2 = img.clone();
                
                if (visualize_batch)
                {
                    auto planner_stats = planner_ptr->GetStats();
                    // cout << "Number of Batches: " << planner_stats.num_evaluated_batches << endl;
                    cout << "Number of Batches evaluated: " << planner_stats.states_eval_.size() << endl;
                    cout << "Number of Lazy Plan found: " << planner_stats.num_lazy_plans << endl;
                    cout << "Single Batche evaluation time: " << planner_stats.cumulative_batch_time/planner_stats.num_evaluated_batches << endl;
                    for (int i = 0; i < planner_stats.num_evaluated_batches; i++)
                    {
                        auto state_vars_vec = planner_stats.states_eval_[i];
                        for (auto state_vars : state_vars_vec)
                        {
                            auto c1 = cv::Point(state_vars[0]-action_params["footprint_size"], state_vars[1]+action_params["footprint_size"]);
                            auto c2 = cv::Point(state_vars[0]+action_params["footprint_size"], state_vars[1]-action_params["footprint_size"]);
                            cv::rectangle(img2, c1, c2, cv::Scalar(0, 255, 255), -1, 8);
                        }
                        auto c1 = cv::Point(starts[exp_idx][0]-action_params["footprint_size"], starts[exp_idx][1]+action_params["footprint_size"]);
                        auto c2 = cv::Point(starts[exp_idx][0]+action_params["footprint_size"], starts[exp_idx][1]-action_params["footprint_size"]);
                        cv::rectangle(img2, c1, c2, cv::Scalar(0, 255, 0), -1, 8);

                        c1 = cv::Point(goals[exp_idx][0]-action_params["footprint_size"], goals[exp_idx][1]+action_params["footprint_size"]);
                        c2 = cv::Point(goals[exp_idx][0]+action_params["footprint_size"], goals[exp_idx][1]-action_params["footprint_size"]);
                        cv::rectangle(img2, c1, c2, cv::Scalar(0, 0, 255), -1, 8);
                        cv::resize(img2, img2, cv::Size(2*img.cols/scale, 2*img.rows/scale));
                        cv::imshow("Plan", img2);
                        // cv::waitKey(500);
                        cv::waitKey(0);
                        img2.setTo(cv::Scalar(0,0,0));
                        img2 = img.clone();
                        // cv::resize(img2, img2, cv::Size(img.cols/(4*scale), img.rows/(4*scale)));
                    }
                }
                
                // Display map with start and goal
                for (auto& plan_element: planner_ptr->GetPlan())
                {
                    auto c1 = cv::Point(plan_element.state_[0]-action_params["footprint_size"], plan_element.state_[1]+action_params["footprint_size"]);
                    auto c2 = cv::Point(plan_element.state_[0]+action_params["footprint_size"], plan_element.state_[1]-action_params["footprint_size"]);
                    cv::rectangle(img2, c1, c2, cv::Scalar(255, 0, 0), -1, 8);
                }

                auto c1 = cv::Point(starts[exp_idx][0]-action_params["footprint_size"], starts[exp_idx][1]+action_params["footprint_size"]);
                auto c2 = cv::Point(starts[exp_idx][0]+action_params["footprint_size"], starts[exp_idx][1]-action_params["footprint_size"]);
                cv::rectangle(img2, c1, c2, cv::Scalar(0, 255, 0), -1, 8);

                c1 = cv::Point(goals[exp_idx][0]-action_params["footprint_size"], goals[exp_idx][1]+action_params["footprint_size"]);
                c2 = cv::Point(goals[exp_idx][0]+action_params["footprint_size"], goals[exp_idx][1]-action_params["footprint_size"]);
                cv::rectangle(img2, c1, c2, cv::Scalar(0, 0, 255), -1, 8);

                cv::resize(img2, img2, cv::Size(2*img.cols/scale, 2*img.rows/scale));
                cv::imshow("Plan", img2);
                cv::waitKey(500);
                // cv::waitKey(0);

                img2.setTo(cv::Scalar(0,0,0));
                cv::imshow("Plan", img2);

            }  
        }
        
        logStats(cost_vec, path_length_vec, num_edges_vec, planner_params["num_threads"], planner_params["heuristic_weight"], action_params["heuristic_noise_factor"], planner_name);

        cout << endl << "************************" << endl;
        cout << "Number of runs: " << num_runs << endl;
        cout << "Mean time: " << accumulate(time_vec.begin(), time_vec.end(), 0.0)/time_vec.size() << endl;
        cout << "Mean cost: " << accumulate(cost_vec.begin(), cost_vec.end(), 0.0)/cost_vec.size() << endl;    
        cout << "Mean threads used: " << accumulate(threads_used_vec.begin(), threads_used_vec.end(), 0.0)/threads_used_vec.size() << "/" << planner_params["num_threads"] << endl;
        cout << "Mean path length: " << accumulate(path_length_vec.begin(), path_length_vec.end(), 0.0)/path_length_vec.size() << endl;
        cout << "Mean expanded states: " << roundOff(accumulate(num_states_vec.begin(), num_states_vec.end(), 0.0)/double(num_states_vec.size()), 2) << endl;
        cout << "Mean evaluated edges: " << roundOff(accumulate(num_edges_vec.begin(), num_edges_vec.end(), 0.0)/double(num_edges_vec.size()), 2) << endl;
        cout << endl << "------------- Mean jobs per thread -------------" << endl;
        for (int tidx = 0; tidx < planner_params["num_threads"]; ++tidx)
        {
            if (num_success != 0)
            {
                cout << "thread: " << tidx << " jobs: " << jobs_per_thread[tidx]/num_success << endl;
            }
        }
        cout << "************************" << endl;
    
        cout << endl << "------------- Mean action eval times -------------" << endl;
        for (auto [action, times] : action_eval_times)
        {
            cout << action << ": " << accumulate(times.begin(), times.end(), 0.0)/times.size() << endl; 
        }
        cout << "************************" << endl;
    }

    cout << endl << "************ Global Stats ************" << endl;
    cout << "Mean time: " << accumulate(all_maps_time_vec.begin(), all_maps_time_vec.end(), 0.0)/all_maps_time_vec.size() << endl;
    cout << "Mean cost: " << accumulate(all_maps_cost_vec.begin(), all_maps_cost_vec.end(), 0.0)/all_maps_cost_vec.size() << endl;    
    cout << "Mean path length: " << accumulate(all_maps_path_length_vec.begin(), all_maps_path_length_vec.end(), 0.0)/all_maps_path_length_vec.size() << endl;
    cout << "Mean expanded states: " << roundOff(accumulate(all_maps_num_states_vec.begin(), all_maps_num_states_vec.end(), 0.0)/double(all_maps_num_states_vec.size()), 2) << endl;
    cout << "Mean evaluated edges: " << roundOff(accumulate(all_maps_num_edges_vec.begin(), all_maps_num_edges_vec.end(), 0.0)/double(all_maps_num_edges_vec.size()), 2) << endl;
    cout << endl << "************************" << endl;

    cout << endl << "------------- Mean action eval times -------------" << endl;
    for (auto [action, times] : all_action_eval_times)
    {
        cout << action << ": " << accumulate(times.begin(), times.end(), 0.0)/times.size() << endl; 
    }
    cout << "************************" << endl;



}
