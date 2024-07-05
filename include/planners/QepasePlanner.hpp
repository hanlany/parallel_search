#ifndef QEPASE_PLANNER_HPP
#define QEPASE_PLANNER_HPP

#include <future>
#include <condition_variable>
#include <planners/GepasePlanner.hpp>

#define INDEPENDENT_CHECK 1

namespace ps
{

class QepasePlanner : public GepasePlanner
{
    public:
        QepasePlanner(ParamsType planner_params);
        ~QepasePlanner();
        bool Plan();

    protected:
        void initialize();
        void expandEdgeLoop(int thread_id);    
        void expandEdge(EdgePtrType edge_ptr, int thread_id);
        void exit();

        void computeDijkstra();

        std::vector<StatePtrType> being_expanded_states_;
        std::vector<std::vector<double>> heuristic_table_;
};

}

#endif
