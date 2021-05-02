#include <pybind11/pybind11.h>

#include <G4ImportanceBiasing.hh>

#include "holder.hh"
#include "typecast.hh"

namespace py = pybind11;

class TRAMPOLINE_NAME(G4ImportanceBiasing) : public G4ImportanceBiasing {
public:
   using G4ImportanceBiasing::G4ImportanceBiasing;

   void ConstructParticle() override { PYBIND11_OVERRIDE(void, G4ImportanceBiasing, ConstructParticle, ); }

   void ConstructProcess() override { PYBIND11_OVERRIDE(void, G4ImportanceBiasing, ConstructProcess, ); }

   TRAMPOLINE_DESTRUCTOR(G4ImportanceBiasing);
};

void export_G4ImportanceBiasing(py::module &m)
{
   py::class_<G4ImportanceBiasing, TRAMPOLINE_NAME(G4ImportanceBiasing), G4VPhysicsConstructor,
              owntrans_ptr<G4ImportanceBiasing>>(m, "G4ImportanceBiasing")

      .def(py::init<const G4String &>(), py::arg("name") = "NoParallelWP")
      .def(py::init<G4GeometrySampler *, const G4String &>(), py::arg("msg"), py::arg("name") = "NoParallelWP")

      .def("ConstructParticle", &G4ImportanceBiasing::ConstructParticle)
      .def("ConstructProcess", &G4ImportanceBiasing::ConstructProcess);
}
