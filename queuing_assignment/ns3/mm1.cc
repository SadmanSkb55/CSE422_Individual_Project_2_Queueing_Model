/* mm1.cc
 * NS-3 M/M/1 queue simulation — event-driven, no TrafficControlHelper.
 * Models arrivals and departures directly using exponential random variables.
 *
 * Build & run (from ns-3 root):
 *   cp scratch/mm1.cc scratch/mm1.cc   (already there)
 *   ./ns3 run scratch/mm1
 *
 * With custom rates:
 *   ./ns3 run "scratch/mm1 --lambda=5 --mu=10"
 *
 * Output (stdout):
 *   n Pn
 *   0 0.0952
 *   1 0.0857
 *   ...
 */

#include "ns3/core-module.h"
#include <iostream>
#include <map>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MM1Queue");

// ── Simulation state ──────────────────────────────────────────────────────────
static int                  g_n           = 0;
static double               g_lastTime    = 0.0;
static std::map<int,double> g_timeIn;
static int                  g_arrivals    = 0;

static double g_lambda      = 9.0;
static double g_mu          = 10.0;
static int    g_maxArrivals = 200000;

void ScheduleArrival();
void ScheduleDeparture();

void RecordAndUpdate(int newN)
{
    double now  = Simulator::Now().GetSeconds();
    g_timeIn[g_n] += now - g_lastTime;
    g_lastTime    = now;
    g_n           = newN;
}

void Arrival()
{
    g_arrivals++;
    RecordAndUpdate(g_n + 1);

    if (g_n == 1)
        ScheduleDeparture();

    if (g_arrivals < g_maxArrivals)
        ScheduleArrival();
    else
        Simulator::Stop();
}

void Departure()
{
    RecordAndUpdate(g_n - 1);
    if (g_n > 0)
        ScheduleDeparture();
}

void ScheduleArrival()
{
    Ptr<ExponentialRandomVariable> erv = CreateObject<ExponentialRandomVariable>();
    erv->SetAttribute("Mean", DoubleValue(1.0 / g_lambda));
    Simulator::Schedule(Seconds(erv->GetValue()), &Arrival);
}

void ScheduleDeparture()
{
    Ptr<ExponentialRandomVariable> erv = CreateObject<ExponentialRandomVariable>();
    erv->SetAttribute("Mean", DoubleValue(1.0 / g_mu));
    Simulator::Schedule(Seconds(erv->GetValue()), &Departure);
}

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.AddValue("lambda",      "Arrival rate",       g_lambda);
    cmd.AddValue("mu",          "Service rate",       g_mu);
    cmd.AddValue("maxArrivals", "Number of arrivals", g_maxArrivals);
    cmd.Parse(argc, argv);

    LogComponentDisableAll(LOG_LEVEL_ALL);

    RngSeedManager::SetSeed(42);
    RngSeedManager::SetRun(1);

    ScheduleArrival();

    Simulator::Run();

    double totalTime = 0.0;
    for (auto &kv : g_timeIn) totalTime += kv.second;

    std::cout << "n Pn" << std::endl;
    for (int n = 0; n <= 30; ++n)
    {
        double pn = g_timeIn.count(n) ? g_timeIn[n] / totalTime : 0.0;
        std::cout << n << " " << pn << std::endl;
    }

    Simulator::Destroy();
    return 0;
}
