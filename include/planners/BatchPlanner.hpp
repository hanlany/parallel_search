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
        void selectBatch();
        void expandState(StatePtrType state_ptr);
        void updateState(StatePtrType& state_ptr, ActionPtrType& action_ptr, ActionSuccessor& action_successor);
        void exit();
        
        // Control variables
        int batch_size_;

        // Data structures
        StateQueueMinType state_open_list_;
        std::vector<StatePrtType> batch_process_list_;
        
        // Multi-threading members
        int num_threads_;
        mutable LockType lock_;

        // Multi-threading members - From Gepase
        // int num_threads_;
        // mutable LockType lock_;
        // mutable std::vector<LockType> lock_vec_; 
        // std::vector<std::future<void>> edge_expansion_futures_;
        // std::vector<EdgePtrType> edge_expansion_vec_;
        // std::vector<int> edge_expansion_status_;
        // std::vector<std::condition_variable> cv_vec_;
        // std::condition_variable cv_;

        // Multi-threading members - From MPLP
        // int num_threads_;
        // mutable LockType lock_;
        // mutable LockType lock_2_;
        // mutable std::vector<LockType> lock_vec_;
        // std::vector<std::future<void>> edge_evaluation_futures_;
        // std::vector<EdgePtrType> edge_evaluation_vec_;
        // std::vector<int> edge_evaluation_status_;
        // std::shared_ptr<std::thread> delegate_edges_process_;
        // std::shared_ptr<std::thread> monitor_paths_process_;

};

}

#endif
