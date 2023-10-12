// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <queue>
#include <cmath>
#include <set>
#include <map>
#include <list>
#define INF 1000000000
namespace py = pybind11;

struct Node {
    Node(int _i = INF, int _j = INF, int _g = 0, int _h = 0) : i(_i), j(_j), g(_g), h(_h), f(_g+_h){}
    int i;
    int j;
    int g;
    int h;
    int f;
    bool operator<(const Node& other) const
    {
        return this->f < other.f or
               (this->f == other.f and (this->g < other.g or
                                       (this->g == other.g and (this->i < other.i or
                                                               (this->i == other.i and this->j < other.j)))));
    }
    bool operator>(const Node& other) const
    {
        return this->f > other.f or
               (this->f == other.f and (this->g > other.g or
                                       (this->g == other.g and (this->i > other.i or
                                                               (this->i == other.i and this->j > other.j)))));
    }
    bool operator==(const Node& other) const
    {
        return this->i == other.i and this->j == other.j;
    }
    bool operator==(const std::pair<int, int> &other) const
    {
        return this->i == other.first and this->j == other.second;
    }
};

class planner {
    std::set<std::pair<int,int>> obstacles;
    std::set<std::pair<int,int>> other_agents;
    std::set<std::pair<int,int>> bad_actions;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> OPEN;
    std::map<std::pair<int, int>, std::pair<int, int>> CLOSED;
    std::pair<int, int> start;
    std::pair<int, int> desired_position;
    std::pair<int, int> goal;
    Node best_node;
    int max_steps;
    inline int h(std::pair<int, int> n)
    {
        return std::abs(n.first - goal.first) + std::abs(n.second - goal.second);
    }
    std::vector<std::pair<int,int>> get_neighbors(std::pair<int, int> node)
    {
        std::vector<std::pair<int,int>> neighbors;
        std::vector<std::pair<int,int>> deltas = {{0,1},{1,0},{-1,0},{0,-1}};
        for(auto d:deltas)
        {
            std::pair<int,int> n(node.first + d.first, node.second + d.second);
            if(obstacles.count(n) == 0)
                neighbors.push_back(n);
        }
        return neighbors;
    }
    void compute_shortest_path()
    {
        Node current;
        int steps = 0;
        while(!OPEN.empty() and steps < max_steps and !(current == goal))
        {
            current = OPEN.top();
            OPEN.pop();
            if(current.h < best_node.h)
                best_node = current;
            steps++;
            for(auto n: get_neighbors({current.i, current.j})) {
                if (CLOSED.find(n) == CLOSED.end() and other_agents.find(n) == other_agents.end()) {
                    OPEN.push(Node(n.first, n.second, current.g + 1, h(n)));
                    CLOSED[n] = {current.i, current.j};
                }
            }
        }
    }
    void reset()
    {
        CLOSED.clear();
        OPEN = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>();
        Node s = Node(start.first, start.second, 0, h(start));
        OPEN.push(s);
        best_node = s;
    }
public:
    planner(int steps=10000) {max_steps = steps;}
    void update_obstacles(const std::list<std::pair<int, int>>& _obstacles,
                          const std::list<std::pair<int, int>>& _other_agents,
                          std::pair<int, int> cur_pos)
    {
        for(auto o:_obstacles)
            obstacles.insert({cur_pos.first + o.first, cur_pos.second + o.second});
        other_agents.clear();
        for(auto o:_other_agents)
            other_agents.insert({cur_pos.first + o.first, cur_pos.second + o.second});
    }
    void update_path(std::pair<int, int> s, std::pair<int, int> g)
    {
        if(desired_position.first < INF and (desired_position.first != s.first or desired_position.second != s.second)) {
            bad_actions.insert(desired_position);
            if (start.first == s.first and start.second == s.second)
                for (auto bad_a: bad_actions)
                    other_agents.insert(bad_a);
        }
        else
            bad_actions.clear();
        start = s;
        goal = g;
        reset();
        compute_shortest_path();
    }
    std::list<std::pair<int, int>> get_path(bool use_best_node = true)
    {
        std::list<std::pair<int, int>> path;
        std::pair<int, int> next_node(INF,INF);
        if(CLOSED.find(goal) != CLOSED.end())
            next_node = goal;
        else if(use_best_node)
            next_node = {best_node.i, best_node.j};
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
        {
            while (CLOSED[next_node] != start) {
                path.push_back(next_node);
                next_node = CLOSED[next_node];
            }
            path.push_back(next_node);
            path.push_back(start);
            path.reverse();
        }
        desired_position = next_node;
        return path;
    }
    std::pair<std::pair<int, int>, std::pair<int, int>> get_next_node(bool use_best_node = true)
    {
        std::pair<int, int> next_node(INF, INF);
        if(CLOSED.find(goal) != CLOSED.end())
            next_node = goal;
        else if(use_best_node)
            next_node = {best_node.i, best_node.j};
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
            while (CLOSED[next_node] != start)
                next_node = CLOSED[next_node];
        if(next_node == start)
            next_node = {INF, INF};
        desired_position = next_node;
        return {start, next_node};
    }
};

PYBIND11_MODULE(planner, m) {
    py::class_<planner>(m, "planner")
            .def(py::init<int>())
            .def("update_obstacles", &planner::update_obstacles)
            .def("update_path", &planner::update_path)
            .def("get_path", &planner::get_path)
            .def("get_next_node", &planner::get_next_node);
}

/*
<%
setup_pybind11(cfg)
%>
*/
