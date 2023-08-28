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
        virtual bool Plan();

    protected:
        void initialize();
        void initializeReplanning();
        void initializeSelect();
        bool replanBPLP();
        void expandState(StatePtrType state_ptr);
        void batchProcess();
        void evaluateEdge(EdgePtrType edge_ptr, int thread_id);
        void evaluateEdgeLoop(int thread_id);
        void constructPlan(StatePtrType goal_state_ptr);
        void monitorPaths();
        void assignEdgePriority(EdgePtrType& edge_ptr);
        void updateEdgePriority(std::vector<EdgePtrType>& edge_ptr, double factor);
        void updateEdgePriority(EdgePtrType& edge_ptr, double factor);
        void exit();

        // Functions not needed
        void delegateEdges();

        StateQueueMinType state_open_list_;
        StateQueueMinType batch_select_list_;
        EdgeQueueMaxType edges_open_;
        std::vector<std::vector<EdgePtrType>> lazy_plans_;
        // -1: no outcome, 1: feasible , 0: infeasible
        std::vector<int> plan_evaluation_outcomes_;
        int successful_plan_idx_;
        double cost_bound_;
        int batch_size_;

        // Multi-threading members
        int num_threads_;
        mutable LockType lock_;
        mutable LockType lock_2_;
        mutable std::vector<LockType> lock_vec_;
        std::vector<std::future<void>> edge_evaluation_futures_;
        std::vector<EdgePtrType> edge_evaluation_vec_;
        std::vector<int> edge_evaluation_status_;
        std::shared_ptr<std::thread> batch_process_;
        std::shared_ptr<std::thread> monitor_paths_process_;

        // Control variables
        std::atomic<bool> terminate_;
        std::atomic<bool> plan_found_;

};

}

#endif
