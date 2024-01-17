#ifndef BATCH_PLANNER_HPP
#define BATCH_PLANNER_HPP

#include <future>
#include <planners/Planner.hpp>

namespace ps
{

class BatchPlanner : public Planner
{
    public:
        BatchPlanner(ParamsType planner_params);
        ~BatchPlanner();
        bool Plan();
        
        // Statistics
        // std::vector<std::vector<StateVarsType>> GetStatesBatches() const {return states_batches_;};

    protected:
        void initialize();
        void initializeSelect();
        void initializeReplanning();
        void expandState(StatePtrType state_ptr);
        void constructPlan(StatePtrType goal_state_ptr);
        void batchProcess();
        void exit();

        StdStateQueueMinType state_open_list_;
        StateQueueHeuristicMinType batch_select_list_;
        int batch_size_;

        // Multi-threading members
        int num_threads_;
        mutable LockType lock_;
        std::shared_ptr<std::thread> batch_process_;

        // Control variables
        std::atomic<bool> terminate_;
        std::atomic<bool> plan_found_;

        // Statistics
        bool visualize_batch_;
        // std::vector<std::vector<StateVarsType>> states_batches_;
};

}

#endif
