#include "GLG4Scint.hh"
#include <G4SteppingManager.hh>
#include <G4OpticalPhysics.hh>
#include <G4EmPenelopePhysics.hh>
#include <G4TrackingManager.hh>
#include <G4TrajectoryContainer.hh>
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4OpticalParameters.hh"
#include <G4Alpha.hh>
#include <G4Neutron.hh>
#include <G4Cerenkov.hh>
#include <G4EmParameters.hh>
#include <G4FastSimulationManagerProcess.hh>
#include <G4HadronicInteractionRegistry.hh>
#include <G4HadronicProcess.hh>
#include <G4Neutron.hh>
#include <G4NeutronHPThermalScattering.hh>
#include <G4NeutronHPThermalScatteringData.hh>
#include <G4OpBoundaryProcess.hh>
#include <G4OpticalPhoton.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTypes.hh>
#include <G4ProcessManager.hh>
#include <G4RunManager.hh>
#include <Shielding.hh>
#include <G4VModularPhysicsList.hh>
#include <G4VUserPhysicsList.hh>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "typecast.hh"
#include "opaques.hh"

namespace py = pybind11;

class ChromaPhysicsList: public Shielding 
{
public:
  explicit ChromaPhysicsList();
  virtual ~ChromaPhysicsList();

  ChromaPhysicsList(const ChromaPhysicsList &)=delete;
  ChromaPhysicsList & operator=(const ChromaPhysicsList &right)=delete;
  void ConstructParticle();
  void ConstructProcess();
  void ConstructOpticalProcesses();
  void AddParameterization();
  void SetCuts();
};

#include <G4UserSteppingAction.hh>
#include <G4UserTrackingAction.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <vector>
#include <map>
#include <iostream>

class Step {
public:
    inline Step(const double _x, const double _y, const double _z, 
                const double _t, 
                const double _dx, const double _dy, const double _dz, 
                const double _ke, const double _edep, const double _qedep, 
                const std::string &_procname) :
                x(_x), y(_y), z(_z), t(_t), dx(_dx), dy(_dy), dz(_dz), 
                ke(_ke), edep(_edep), qedep(_qedep), procname(_procname) {
    }
    
    inline ~Step() { }

    const double x,y,z,t,dx,dy,dz,ke,edep,qedep;
    const std::string procname;
};

class Track {
public:
    inline Track() : id(-1) {
        this->steps = new std::vector<Step>();
        this->children = new std::vector<int>();
    }
    inline ~Track() {
        delete this->steps;
        delete this->children;
    }
    
    int id, parent_id, pdg_code;
    double weight;
    std::string name;
    std::vector<Step>* steps;
    std::vector<int>* children;
    
    void appendStepPoint(const G4StepPoint* point, const G4Step* step, double qedep, const bool initial = false);  
    inline std::vector<Step>* getSteps() { return this->steps; };  
    long unsigned int getNumSteps();
    
    inline long unsigned int getNumChildren() { return children->size(); }
    inline int getChildTrackID(long unsigned int i) {
        if( i <= this->children->size() ){
            return this->children->at(i);
        } else {
            return 0;
        }
    }
    inline void addChild(int trackid) { children->push_back(trackid); }
};

class SteppingAction : public G4UserSteppingAction
{
public:
  SteppingAction();
  virtual ~SteppingAction();
  
  void EnableScint(bool enabled);
  void EnableTracking(bool enabled);
  
  void UserSteppingAction(const G4Step* aStep);
  
  void ClearTracking();
  Track* getTrack(int id);
  
private:
    bool scint;
    
    bool tracking;
    bool children_mapped;
    void mapChildren();
    std::map<int,Track*> trackmap;

};

class TrackingAction : public G4UserTrackingAction
{
public:
  TrackingAction();
  virtual ~TrackingAction();
  
  int GetNumPhotons() const;
  void Clear();
  
  void GetX(double *x) const;
  void GetY(double *y) const;
  void GetZ(double *z) const;
  void GetDirX(double *dir_x) const;
  void GetDirY(double *dir_y) const;
  void GetDirZ(double *dir_z) const;
  void GetPolX(double *pol_x) const;
  void GetPolY(double *pol_y) const;
  void GetPolZ(double *pol_z) const;

  void GetWavelength(double *wl) const;
  void GetT0(double *t) const;
  
  void GetParentTrackID(int *t) const;
  void GetFlags(uint32_t *flags) const;

  virtual void PreUserTrackingAction(const G4Track *);

protected:
  std::vector<G4ThreeVector> pos;
  std::vector<G4ThreeVector> dir;
  std::vector<G4ThreeVector> pol;
  std::vector<int> parentTrackID;
  std::vector<uint32_t> flags;
  std::vector<double> wavelength;
  std::vector<double> t0;
};

ChromaPhysicsList::ChromaPhysicsList() : Shielding(){
}

void ChromaPhysicsList::ConstructParticle() {
  Shielding::ConstructParticle();
  G4OpticalPhoton::OpticalPhotonDefinition();
}

void ChromaPhysicsList::ConstructProcess() {
  G4EmParameters *param = G4EmParameters::Instance();
  param->SetStepFunctionLightIons(0.01, 0.01);
  //param->SetStepFunctionMuHad(this->stepRatioMuHad, this->finalRangeMuHad);

  AddParameterization();
  Shielding::ConstructProcess();
  ConstructOpticalProcesses();
  //EnableThermalNeutronScattering();
}

void ChromaPhysicsList::ConstructOpticalProcesses() {
  bool enableCherenkov = true;
  bool enableScintillation = true;
  // Cherenkov: default G4Cerenkov
  //
  // Request that Cerenkov photons be tracked first, before continuing
  // originating particle step.  Otherwise, we get too many secondaries!
  G4Cerenkov *cerenkovProcess = nullptr;
  //if (this->IsCerenkovEnabled) {
  if (true) {
    cerenkovProcess = new G4Cerenkov();
    cerenkovProcess->SetTrackSecondariesFirst(true);
    cerenkovProcess->SetMaxNumPhotonsPerStep(1);
  }

  // Attenuation: RAT's GLG4OpAttenuation
  //
  // GLG4OpAttenuation implements Rayleigh scattering.
  //GLG4OpAttenuation *attenuationProcess = new GLG4OpAttenuation();

  // Scintillation: RAT's GLG4Scint
  //
  // Create three scintillation processes which depend on the mass.
  G4double protonMass = G4Proton::Proton()->GetPDGMass();
  G4double alphaMass = G4Alpha::Alpha()->GetPDGMass();
  GLG4Scint *defaultScintProcess = new GLG4Scint();
  GLG4Scint *nucleonScintProcess = new GLG4Scint("nucleon", 0.9 * protonMass);
  GLG4Scint *alphaScintProcess = new GLG4Scint("alpha", 0.9 * alphaMass);

  // Optical boundary processes: default G4
  G4OpBoundaryProcess *opBoundaryProcess = new G4OpBoundaryProcess();
  // Rayleigh Scattering
  //OpRayleigh *opRayleigh = new OpRayleigh();

  cerenkovProcess->SetVerboseLevel(verboseLevel - 1);
  //defaultScintProcess->SetVerboseLevel(verboseLevel - 1);
  //nucleonScintProcess->SetVerboseLevel(verboseLevel - 1);
  //alphaScintProcess->SetVerboseLevel(verboseLevel - 1);
  opBoundaryProcess->SetVerboseLevel(verboseLevel - 1);

  // Apply processes to all particles where applicable
  GetParticleIterator()->reset();
  while ((*GetParticleIterator())()) {
    G4ParticleDefinition *particle = GetParticleIterator()->value();
    G4ProcessManager *pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();
    if (cerenkovProcess->IsApplicable(*particle)) {
      pmanager->AddProcess(cerenkovProcess);
      pmanager->SetProcessOrdering(cerenkovProcess, idxPostStep);
    }
    if (particleName == "opticalphoton") {
      //pmanager->AddDiscreteProcess(attenuationProcess);
      pmanager->AddDiscreteProcess(opBoundaryProcess);
      //pmanager->AddDiscreteProcess(opRayleigh);
    }
  }
}

void ChromaPhysicsList::AddParameterization() {
  G4FastSimulationManagerProcess *fastSimulationManagerProcess = new G4FastSimulationManagerProcess();
  GetParticleIterator()->reset();
  while ((*GetParticleIterator())()) {
    G4ParticleDefinition *particle = GetParticleIterator()->value();
    G4ProcessManager *pmanager = particle->GetProcessManager();
    if (particle->GetParticleName() == "opticalphoton") {
      pmanager->AddProcess(fastSimulationManagerProcess, -1, -1, 1);
    }
  }
}

ChromaPhysicsList::~ChromaPhysicsList()
{
}

void ChromaPhysicsList::SetCuts(){
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets 
  //   the default cut value for all particle types 
  SetCutsWithDefault();   
}

SteppingAction::SteppingAction()
{
    scint = true;
    tracking = false;
    children_mapped = false;
}

SteppingAction::~SteppingAction()
{
}

void SteppingAction::EnableScint(bool enabled) {
    scint = enabled;
}


void SteppingAction::EnableTracking(bool enabled) {
    tracking = enabled;
}


void SteppingAction::UserSteppingAction(const G4Step *step) {
    double qedep = 0.0;
    try {
        qedep = step->GetTotalEnergyDeposit();
    } catch (...) {
        std::cout << "SteppingAction::UserSteppingAction: GetTotalEnergyDeposit failed" << std::endl;
    }

    if (scint) {
        //std::cout << "Scint true" << std::endl;
        G4VParticleChange * pParticleChange = GLG4Scint::GenericPostPostStepDoIt(step);
        
        if (pParticleChange) {
            qedep = GLG4Scint::GetLastEdepQuenched();
            
            const size_t nsecondaries = pParticleChange->GetNumberOfSecondaries();
            
            for (size_t i = 0; i < nsecondaries; i++) { 
                G4Track * tempSecondaryTrack = pParticleChange->GetSecondary(i);
                fpSteppingManager->GetfSecondary()->push_back( tempSecondaryTrack );
            }
            
            pParticleChange->Clear();
        }
    }
    
    if (tracking) {
        const G4Track *g4track = step->GetTrack();
        const int trackid = g4track->GetTrackID();
        if( trackmap.find(trackid) == trackmap.end() ) {
            trackmap[trackid] = new Track();
        }
        Track* track = trackmap.find(trackid)->second;
        if (track->id == -1) {
            // How can these two be different?
            //std::cout << "SteppingAction::UserSteppingAction: new track with id: " << trackid << std::endl;
            track->id = trackid;
            track->parent_id = g4track->GetParentID();
            auto particle_definition = g4track->GetDefinition();
            track->pdg_code = particle_definition->GetPDGEncoding();
            track->weight = g4track->GetWeight();
            track->name = particle_definition->GetParticleName();
            track->appendStepPoint(step->GetPreStepPoint(), step, 0.0, true);
        }
        track->appendStepPoint(step->GetPostStepPoint(), step, qedep);
    }
}


void SteppingAction::ClearTracking() {
    //std::cout << "SteppingAction::ClearTracking" << std::endl;
    trackmap.clear();    
    children_mapped = false;
}

Track* SteppingAction::getTrack(int id) {
    if (!children_mapped) mapChildren();
    if( trackmap.find(id) == trackmap.end() ){
        std::cout << "Track " << id << " not found ... this should not happen" << std::endl;
        std::cout << "trackmap.size() = " << trackmap.size() << std::endl;
    }
    return trackmap.find(id)->second;
}

void SteppingAction::mapChildren() {
    for (auto it = trackmap.begin(); it != trackmap.end(); it++) {
        const int parent = it->second->parent_id;
        // Todo: This leads to the segfault. Which is obvious if the parent does not exist,
        // then how can it have a child? For some reason parent 0 never makes it, creating an orphan.
        if( trackmap.find(parent) == trackmap.end() ){
            std::cout << "Parent " << parent << " not found ... " << it->second->id << " is an orphan." << std::endl;
        } else {
            trackmap[parent]->addChild(it->first);
        }
    }
    children_mapped = true;
}

long unsigned int Track::getNumSteps() { 
    return this->steps->size(); 
}  

void Track::appendStepPoint(const G4StepPoint* point, const G4Step* step, double qedep, const bool initial) {
    auto step_status = point->GetStepStatus();
    const double len = initial ? 0.0 : step->GetStepLength();
    
    const G4ThreeVector &position = point->GetPosition();
    const double x = position.x();
    const double y = position.y();
    const double z = position.z();
    const double t = point->GetGlobalTime();

    const G4ThreeVector &direction = point->GetMomentumDirection();
    const double dx = direction.x();
    const double dy = direction.y();
    const double dz = direction.z();
    const double ke = point->GetKineticEnergy();
    const double edep = step->GetTotalEnergyDeposit();

    const G4VProcess *process = point->GetProcessDefinedStep();
    std::string procname;
    auto step_track = step->GetTrack();
    if (process) {
        procname = process->GetProcessName();
    } else if (step_track->GetCreatorProcess()) {
        auto creator_process = step_track->GetCreatorProcess();
        procname = creator_process->GetProcessName();
    } else {
        procname = "---";
    }
    
    Step new_step = Step(x,y,z,t,dx,dy,dz,ke,edep,qedep,procname);
    this->steps->push_back(new_step);
}

TrackingAction::TrackingAction() {
}

TrackingAction::~TrackingAction() {
}

int TrackingAction::GetNumPhotons() const {
    return pos.size();
}

void TrackingAction::Clear() {
    pos.clear();
    dir.clear();
    pol.clear();
    wavelength.clear();
    t0.clear();
    parentTrackID.clear();
    flags.clear();
}

void TrackingAction::PreUserTrackingAction(const G4Track *track) {
    G4ParticleDefinition *particle = track->GetDefinition();
    std::string print_statement = "Tracking ... --->>> " + particle->GetParticleName();
    if (particle->GetParticleName() == "opticalphoton") {
        std::string print_statement = "TRACKING !!!! --->>> " + particle->GetParticleName();
        uint32_t flag = 0;
        auto creator_process = track->GetCreatorProcess();
        G4String process = (creator_process) ? creator_process->GetProcessName() : "Unknown";
        print_statement += "~ process: " + process;
        if( process == "Scintillation" ){
            flag |= 1 << 11; //see chroma/cuda/photons.h
        } else if( process == "Cerenkov" ) {
            flag |= 1 << 10; //see chroma/cuda/photons.h
        }
        flags.push_back(flag);
        pos.push_back(track->GetPosition()/mm);
        dir.push_back(track->GetMomentumDirection());
        pol.push_back(track->GetPolarization());
        wavelength.push_back( (h_Planck * c_light / track->GetKineticEnergy()) / nanometer );
        t0.push_back(track->GetGlobalTime() / ns);
        parentTrackID.push_back(track->GetParentID());
        const_cast<G4Track *>(track)->SetTrackStatus(fStopAndKill);
    }
}

#define PhotonCopy(type,name,accessor) \
void TrackingAction::name(type *arr) const { \
    for (unsigned i=0; i < pos.size(); i++) arr[i] = accessor; \
}
    
PhotonCopy(double,GetX,pos[i].x())
PhotonCopy(double,GetY,pos[i].y())
PhotonCopy(double,GetZ,pos[i].z())
PhotonCopy(double,GetDirX,dir[i].x())
PhotonCopy(double,GetDirY,dir[i].y())
PhotonCopy(double,GetDirZ,dir[i].z())
PhotonCopy(double,GetPolX,pol[i].x())
PhotonCopy(double,GetPolY,pol[i].y())
PhotonCopy(double,GetPolZ,pol[i].z())
PhotonCopy(double,GetWavelength,wavelength[i])
PhotonCopy(double,GetT0,t0[i])
PhotonCopy(uint32_t,GetFlags,flags[i])
PhotonCopy(int,GetParentTrackID,parentTrackID[i])

class PyChromaPhysicsList : public ChromaPhysicsList, public py::trampoline_self_life_support {
  public:
    using ChromaPhysicsList::ChromaPhysicsList;
    void SetCuts() override { PYBIND11_OVERRIDE(void, ChromaPhysicsList, SetCuts, ); }
};

template <typename T, void (TrackingAction::*Method)(T*) const>
py::array_t<T> PhotonAccessor(const TrackingAction *pta) {
  py::array_t<T> r(pta->GetNumPhotons());
  (pta->*Method)((T*)r.request().ptr );
  return r;
}

template <typename T, const T (Step::*Method)>
py::array_t<T> StepAccessor(Track *pta) {
    std::vector<Step>* steps = pta->getSteps();
    py::array_t<T> r(steps->size());
    T* np_ptr = (T*)r.request().ptr;
    for (size_t i=0; i < steps->size(); i++){
        auto step = steps->at(i);
        np_ptr[i] = step.*Method;
    }
    return r;
}

void export_ChromaPhysicsList(py::module &m)
{
  py::class_<ChromaPhysicsList, Shielding>(m, "ChromaPhysicsList", "ChromaPhysicsList").def(py::init<>());

  py::class_<Track>(m, "Track")
    .def(py::init<>())
    .def_readonly("track_id",&Track::id)
    .def_readonly("parent_track_id",&Track::parent_id)
    .def_readonly("pdg_code",&Track::pdg_code)
    .def_readonly("weight",&Track::weight)
    .def_readonly("name",&Track::name)
    .def("getNumSteps",&Track::getNumSteps)
    .def("getStepX",StepAccessor<double, &Step::x>)
    .def("getStepY",StepAccessor<double, &Step::y>)
    .def("getStepZ",StepAccessor<double, &Step::z>)
    .def("getStepT",StepAccessor<double, &Step::t>)
    .def("getStepDX",StepAccessor<double, &Step::dx>)
    .def("getStepDY",StepAccessor<double, &Step::dy>)
    .def("getStepDZ",StepAccessor<double, &Step::dz>)
    .def("getStepKE",StepAccessor<double, &Step::ke>)
    .def("getStepEDep",StepAccessor<double, &Step::edep>)
    .def("getStepQEDep",StepAccessor<double, &Step::qedep>)
    .def("getNumChildren",&Track::getNumChildren)
    .def("getChildTrackID",&Track::getChildTrackID);
  
  py::class_<TrackingAction, G4UserTrackingAction>(m, "TrackingAction")
    .def(py::init<>())
    .def("GetNumPhotons", &TrackingAction::GetNumPhotons)
    .def("Clear", &TrackingAction::Clear)
    .def("GetX", PhotonAccessor<double, &TrackingAction::GetX>)
    .def("GetY", PhotonAccessor<double, &TrackingAction::GetY>)
    .def("GetZ", PhotonAccessor<double, &TrackingAction::GetZ>)
    .def("GetDirX", PhotonAccessor<double, &TrackingAction::GetDirX>)
    .def("GetDirY", PhotonAccessor<double, &TrackingAction::GetDirY>)
    .def("GetDirZ", PhotonAccessor<double, &TrackingAction::GetDirZ>)
    .def("GetPolX", PhotonAccessor<double, &TrackingAction::GetPolX>)
    .def("GetPolY", PhotonAccessor<double, &TrackingAction::GetPolY>)
    .def("GetPolZ", PhotonAccessor<double, &TrackingAction::GetPolZ>)
    .def("GetWavelength", PhotonAccessor<double, &TrackingAction::GetWavelength>)
    .def("GetT0", PhotonAccessor<double, &TrackingAction::GetT0>)
    .def("GetParentTrackID", PhotonAccessor<int, &TrackingAction::GetParentTrackID>)
    .def("GetFlags", PhotonAccessor<uint32_t, &TrackingAction::GetFlags>);

  py::class_<SteppingAction, G4UserSteppingAction>(m, "SteppingAction")
    .def(py::init<>())
    .def("EnableScint",&SteppingAction::EnableScint)
    .def("EnableTracking",&SteppingAction::EnableTracking)
    .def("ClearTracking",&SteppingAction::ClearTracking)
    //.def("getTrack",&SteppingAction::getTrack,return_value_policy<reference_existing_object>())
    .def("getTrack",&SteppingAction::getTrack);
}

void export_Chroma(py::module &m)
{
    export_ChromaPhysicsList(m);
}
