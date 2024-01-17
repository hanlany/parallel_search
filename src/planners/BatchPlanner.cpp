#include <iostream>
#include <algorithm>
#include <planners/BatchPlanner.hpp>

using namespace std;
using namespace ps;

#define UPDATE_BATCH_Q 0

BatchPlanner::BatchPlanner(ParamsType planner_params):
Planner(planner_params)
{    
    num_threads_  = planner_params["num_threads"];
    batch_size_ = planner_params["batch_size"];
    visualize_batch_ = planner_params["visualize_batch"];
}

BatchPlanner::~BatchPlanner()
{
    
}

bool BatchPlanner::Plan()
{
    initialize();
    startTimer();
    
    // Initialize a priority queue for batch process
    initializeSelect();

    // Spawn Batch processing thread
    batch_process_ = shared_ptr<thread>(new thread(&BatchPlanner::batchProcess, this));
 

    // Working Loop
    while (!terminate_)
    {
        while(!state_open_list_.empty())
        {
            auto state_ptr = state_open_list_.top();
            state_open_list_.pop();

            // Hit goal region
            if (isGoalState(state_ptr))
            {
                planner_stats_.num_lazy_plans++;
                constructPlan(state_ptr);
                if (plan_found_)
                {
                    terminate_ = true;
                    break;
                }
                else
                {
                    // Reset OPEN and CLOSED
                    initializeReplanning();
                    continue;
                }
            }
            
            // Expand state condition check
            if (state_ptr->IsVisited())
            {
                continue;
            }
            
            expandState(state_ptr);
            state_ptr->SetVisited();
        }
    }

    terminate_ = true;

    auto t_end = chrono::steady_clock::now();
    double t_elapsed = chrono::duration_cast<chrono::nanoseconds>(t_end-t_start_).count();
    planner_stats_.total_time = 1e-9*t_elapsed;

    exit();

    return plan_found_;
}

void BatchPlanner::initialize()
{
    Planner::initialize();
    planner_stats_.num_jobs_per_thread.resize(num_threads_, 0);

    terminate_ = false;
    plan_found_ = false;

    state_open_list_.push(start_state_ptr_);
    
    // Stats
    // states_batches_.clear();
}

void BatchPlanner::initializeReplanning()
{
    resetStates();

    while (!state_open_list_.empty())
    {
        state_open_list_.pop();
    }
    
    start_state_ptr_->SetGValue(0);
    state_open_list_.push(start_state_ptr_);
}


void BatchPlanner::expandState(StatePtrType state_ptr)
{
    // TODO: Rewrite logic checks for cleaner code
    if (VERBOSE) state_ptr->Print("Expanding");
    // return;
    
    bool state_expanded_before = false;
    for (auto& action_ptr: actions_ptrs_)
    {
        EdgePtrType edge_ptr = NULL;
        edge_ptr = new Edge(state_ptr, action_ptr);
        auto edge_key = getEdgeKey(edge_ptr);
        delete edge_ptr;
        edge_ptr = NULL;
        // Don't need a lock since no other thread is adding to edge_map_ except this. Which means
        // that when this line is being executed, no thread is modifying (writing) edge_map_.
        auto it_edge = edge_map_.find(edge_key); 

        bool edge_generated = true;

        if (it_edge == edge_map_.end())
        {
            if (VERBOSE) cout << "Expand: Edge not generated " << endl;
            edge_generated = false;
        }
        else
        {
            if (VERBOSE) cout << "Expand: Edge already exists" << endl;
            edge_ptr = it_edge->second;
        }
        // else if (it_edge->second->is_closed_)
        // {
        //     // Edge in Eclosed
        //     if (VERBOSE) cout << "Expand: Edge in closed " << endl;
        //     edge_ptr = it_edge->second;
        // }
        // else if (edges_open_.contains(it_edge->second))
        // {
        //     // Edge in Eopen
        //     if (VERBOSE) cout << "Expand: Edge in open " << endl;
        //     edge_ptr = it_edge->second;
        // }
        // else if (it_edge->second->is_eval_)
        // {
        //     // Edge in Eeval
        //     if (VERBOSE) cout << "Expand: Edge in eval " << endl;
        //     edge_ptr = it_edge->second;
        // }
        // else if (it_edge->second->is_invalid_)
        // {
        //     if (VERBOSE) cout << "Invalid edge" << endl;
        //     continue;
        // }            
        //

        if (!edge_generated && action_ptr->CheckPreconditions(state_ptr->GetStateVars()))
        {

            // Evaluate the edge lazily
            auto t_start = chrono::steady_clock::now();
            auto action_successor = action_ptr->GetSuccessorProxy(state_ptr->GetStateVars());
            auto t_end = chrono::steady_clock::now();

            // This is always success in our case
            if (action_successor.success_)
            {
                auto successor_state_ptr = constructState(action_successor.successor_state_vars_costs_.back().first);
                edge_ptr = new Edge(state_ptr, successor_state_ptr, action_ptr);
                edge_ptr->SetCost(action_successor.successor_state_vars_costs_.back().second);
                edge_map_.insert(make_pair(getEdgeKey(edge_ptr), edge_ptr));
            }
        }

        // lock_.lock();
        // if (!(edge_ptr->child_state_ptr_->IsValid()))
        // {
        //     delete edge_ptr;
        //     edge_ptr = NULL;
        // }
        // else
        // {
        //     // lock_.lock();
        //     // edges_open_.push(edge_ptr);
        //     // lock_.unlock();
        // }
        // lock_.unlock();

        if (edge_ptr && edge_ptr->child_state_ptr_ )
        {
            auto successor_state_ptr = edge_ptr->child_state_ptr_;
            double new_g_val = state_ptr->GetGValue() + edge_ptr->GetCost();
            
            if (successor_state_ptr->GetGValue() > new_g_val)
            {

                double h_val = successor_state_ptr->GetHValue();

                if (h_val == -1)
                {
                    h_val = computeHeuristic(successor_state_ptr);
                    successor_state_ptr->SetHValue(h_val);        
                }

                if (h_val != DINF)
                {
                    if (!edge_ptr->child_state_ptr_->IsVisited()) 
                    {
                        lock_.lock();
                        bool child_valid = edge_ptr->child_state_ptr_->IsValid();
                        lock_.unlock();
                        if (child_valid)
                        {
                            h_val_min_ = h_val < h_val_min_ ? h_val : h_val_min_;
                            successor_state_ptr->SetGValue(new_g_val);
                            successor_state_ptr->SetFValue(new_g_val + heuristic_w_*h_val);
                            successor_state_ptr->SetIncomingEdgePtr(edge_ptr);

                            // if (VERBOSE) successor_state_ptr->Print("Pushing Successor");
                            // cout << state_open_list_.size() << endl;
                            // cout << state_open_list_.empty() << endl;
                            
                            // BUG: smpl heap element can only manage 1 instrusive heap 
                            // TMP solution: Use std priority queue for state open list
 
                            // if (state_open_list_.empty())
                            // {
                            //     state_open_list_.push(successor_state_ptr);
                            // }
                            // else if (state_open_list_.contains(successor_state_ptr))
                            // {
                            //     state_open_list_.decrease(successor_state_ptr);
                            // }
                            // else
                            // {
                                // state_open_list_.push(successor_state_ptr);
                            // }
                            state_open_list_.push(successor_state_ptr);
                        }
                    }
                }
            }
        }
        // else
        // {
        //     // Insert into Einvalid if no valid successor is generated
        //     edge_ptr = new Edge(state_ptr, action_ptr);
        //     edge_ptr->SetCost(DINF);
        //     edge_ptr->is_invalid_ = true;
        //
        //     // Insert invalid edge into edge_map_. edge_map_ insert has to be under lock because edge_map_.find is happening
        //     // in updateEdgeCbk
        //     // lock_.lock();
        //     edge_map_.insert(make_pair(getEdgeKey(edge_ptr), edge_ptr));
        //     // lock_.unlock();
        // }    

        if (edge_generated)
            state_expanded_before = true;

    }

    // if (state_expanded_before)
    // {
    //     num_reexpansions += 1;
    //     t_reexpansion_ += 1e-9*t_elapsed_expand;
    // }
    // else
    // {
    //     num_firstexpansions += 1;
    //     t_firstexpansion_ += 1e-9*t_elapsed_expand;
    // }
    // }

}

void BatchPlanner::batchProcess()
{
    while (!terminate_)
    {
        lock_.lock();
        // Select Top K states to validate
        vector<StatePtrType> states_to_validate_ptr;
        vector<StateVarsType> states_to_validate_var;
        for (int i = 0; i < batch_size_; ++i)
        {
            if (batch_select_list_.size() == 0)
                break;
            states_to_validate_ptr.push_back(batch_select_list_.min());
            batch_select_list_.min()->SetBeingExpanded();
            states_to_validate_var.push_back(batch_select_list_.min()->GetStateVars());
            batch_select_list_.pop();
        }
        lock_.unlock();
        
        
        if (states_to_validate_ptr.size() == 0)
        {
            // Out of states to validate
            continue;
        }

        if (visualize_batch_)
        {
            planner_stats_.states_eval_.push_back(states_to_validate_var);
        }
        
        planner_stats_.num_evaluated_batches++;    


        auto t_start = chrono::steady_clock::now();
        // Batch validate states
        auto any_action_ptr = actions_ptrs_[0];
        vector<bool> state_validity(states_to_validate_ptr.size(), false);
        state_validity = any_action_ptr->StateValidateBatch(states_to_validate_var, 0);
        auto t_end = chrono::steady_clock::now();
        
        double t_elapsed = chrono::duration_cast<chrono::nanoseconds>(t_end-t_start).count();            
        planner_stats_.cumulative_batch_time += 1e-9*t_elapsed;

        lock_.lock();
        for (int i = 0; i < states_to_validate_ptr.size(); i++)
        {
            states_to_validate_ptr[i]->SetEvaluated();
            states_to_validate_ptr[i]->UnsetBeingExpanded();
            if (!state_validity[i])
                states_to_validate_ptr[i]->UnsetValid();
        }
        lock_.unlock();
    }
}

void BatchPlanner::initializeSelect()
{
    // For now, do it explicitly
    vector<StateVarsType> all_states_vars;
    all_states_vars = explicit_graph_generator_(start_state_ptr_->GetStateVars());
    // Construct all states and push them into the batch_select_list_
    for (auto& state_vars: all_states_vars)
    {
        StatePtrType state_ptr = constructState(state_vars);
        // For now use H(start,curr) + H(goal)
        state_ptr->SetBValue(computeHeuristic(start_state_ptr_, state_ptr) + computeHeuristic(state_ptr));
        batch_select_list_.push(state_ptr);
    }
    // cout << "Batch_select_list_ size: " << batch_select_list_.size() << endl;
    // auto top_state_ptr_ = batch_select_list_.min();
    // cout << "Top state's priority: " << top_state_ptr_->GetBValue() << endl;
}

void BatchPlanner::constructPlan(StatePtrType goal_state_ptr)
{
    if (VERBOSE) goal_state_ptr->Print("Find Goal");
    auto state_ptr = goal_state_ptr;
    vector<StatePtrType> states_in_plan;
    vector<PlanElement> plan;
    bool plan_valid = true;
    double cost = 0;
    
    // Get all states in the path
    while(state_ptr)
    {
        if (state_ptr->GetIncomingEdgePtr())
        {
            plan.insert(plan.begin(), PlanElement(state_ptr->GetStateVars(), state_ptr->GetIncomingEdgePtr()->action_ptr_, state_ptr->GetIncomingEdgePtr()->GetCost()));
            states_in_plan.insert(states_in_plan.begin(), state_ptr);
            cost += state_ptr->GetIncomingEdgePtr()->GetCost();
            state_ptr = state_ptr->GetIncomingEdgePtr()->parent_state_ptr_;
        }
        else
        {
            // For start state_ptr, there is no incoming edge
            plan.insert(plan.begin(), PlanElement(state_ptr->GetStateVars(), NULL, 0));
            states_in_plan.insert(states_in_plan.begin(), state_ptr);
            state_ptr = NULL;
        }
    }
    
    // Check all states in the path
    lock_.lock();
    for (auto& state_ptr: states_in_plan)
    {
        if (VERBOSE) state_ptr->Print("Constructing Plan");
        if (!state_ptr->IsEvaluated())
        {
            plan_valid = false;
            if (batch_select_list_.contains(state_ptr))
            {
                state_ptr->SetBValue(-1);
                batch_select_list_.decrease(state_ptr);
            }
            continue;
        }
        if (!state_ptr->IsValid())
        {
            plan_valid = false;
        }
    }
    lock_.unlock();
    
    if (plan_valid)
    {
        plan_found_ = true;
        goal_state_ptr_ = goal_state_ptr;
        plan_.swap(plan);
        planner_stats_.path_cost= cost;
        planner_stats_.path_length = plan_.size();
    }
}

void BatchPlanner::exit()
{
    batch_process_->join();

    // Clear open list
    while (!state_open_list_.empty())
    {
        state_open_list_.pop();
    }
    
    // Clear batch select list
    while (!batch_select_list_.empty())
    {
        batch_select_list_.pop();
    }

    Planner::exit();
}
