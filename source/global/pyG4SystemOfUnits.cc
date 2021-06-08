#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <G4SystemOfUnits.hh>

namespace py = pybind11;

void export_G4SystemOfUnits(py::module &m)
{
   m.attr("ampere")           = ampere;
   m.attr("angstrom")         = angstrom;
   m.attr("atmosphere")       = atmosphere;
   m.attr("bar")              = bar;
   m.attr("barn")             = barn;
   m.attr("becquerel")        = becquerel;
   m.attr("candela")          = candela;
   m.attr("centimeter")       = centimeter;
   m.attr("centimeter2")      = centimeter2;
   m.attr("centimeter3")      = centimeter3;
   m.attr("cL")               = cL;
   m.attr("cm")               = cm;
   m.attr("cm2")              = cm2;
   m.attr("cm3")              = cm3;
   m.attr("coulomb")          = coulomb;
   m.attr("curie")            = curie;
   m.attr("deg")              = deg;
   m.attr("degree")           = degree;
   m.attr("dL")               = dL;
   m.attr("e_SI")             = e_SI;
   m.attr("electronvolt")     = electronvolt;
   m.attr("eplus")            = eplus;
   m.attr("eV")               = eV;
   m.attr("farad")            = farad;
   m.attr("fermi")            = fermi;
   m.attr("g")                = g;
   m.attr("gauss")            = gauss;
   m.attr("GeV")              = GeV;
   m.attr("gigaelectronvolt") = gigaelectronvolt;
   m.attr("gram")             = gram;
   m.attr("gray")             = gray;
   m.attr("henry")            = henry;
   m.attr("hep_pascal")       = hep_pascal;
   m.attr("pascal")           = hep_pascal;
   m.attr("hertz")            = hertz;
   m.attr("joule")            = joule;
   m.attr("kelvin")           = kelvin;
   m.attr("keV")              = keV;
   m.attr("kg")               = kg;
   m.attr("kiloelectronvolt") = kiloelectronvolt;
   m.attr("kilogauss")        = kilogauss;
   m.attr("kilogram")         = kilogram;
   m.attr("kilohertz")        = kilohertz;
   m.attr("kilometer")        = kilometer;
   m.attr("kilometer2")       = kilometer2;
   m.attr("kilometer3")       = kilometer3;
   m.attr("kilovolt")         = kilovolt;
   m.attr("km")               = km;
   m.attr("km2")              = km2;
   m.attr("km3")              = km3;
   m.attr("L")                = L;
   m.attr("liter")            = liter;
   m.attr("lumen")            = lumen;
   m.attr("lux")              = lux;
   m.attr("m")                = CLHEP::m;
   m.attr("m2")               = m2;
   m.attr("m3")               = m3;
   m.attr("megaelectronvolt") = megaelectronvolt;
   m.attr("megahertz")        = megahertz;
   m.attr("megavolt")         = megavolt;
   m.attr("meter")            = meter;
   m.attr("meter2")           = meter2;
   m.attr("meter3")           = meter3;
   m.attr("MeV")              = MeV;
   m.attr("mg")               = mg;
   m.attr("microampere")      = microampere;
   m.attr("microbarn")        = microbarn;
   m.attr("microfarad")       = microfarad;
   m.attr("micrometer")       = micrometer;
   m.attr("microsecond")      = microsecond;
   m.attr("milliampere")      = milliampere;
   m.attr("millibarn")        = millibarn;
   m.attr("millifarad")       = millifarad;
   m.attr("milligram")        = milligram;
   m.attr("millimeter")       = millimeter;
   m.attr("millimeter2")      = millimeter2;
   m.attr("millimeter3")      = millimeter3;
   m.attr("milliradian")      = milliradian;
   m.attr("millisecond")      = millisecond;
   m.attr("mL")               = mL;
   m.attr("mm")               = mm;
   m.attr("mm2")              = mm2;
   m.attr("mm3")              = mm3;
   m.attr("mole")             = mole;
   m.attr("mrad")             = mrad;
   m.attr("ms")               = ms;
   m.attr("nanoampere")       = nanoampere;
   m.attr("nanobarn")         = nanobarn;
   m.attr("nanofarad")        = nanofarad;
   m.attr("nanometer")        = nanometer;
   m.attr("nanosecond")       = nanosecond;
   m.attr("newton")           = newton;
   m.attr("nm")               = nm;
   m.attr("ns")               = ns;
   m.attr("ohm")              = ohm;
   m.attr("parsec")           = parsec;
   m.attr("pc")               = pc;
   m.attr("perCent")          = perCent;
   m.attr("perMillion")       = perMillion;
   m.attr("perThousand")      = perThousand;
   m.attr("petaelectronvolt") = petaelectronvolt;
   m.attr("PeV")              = PeV;
   m.attr("picobarn")         = picobarn;
   m.attr("picofarad")        = picofarad;
   m.attr("picosecond")       = picosecond;
   m.attr("ps")               = ps;
   m.attr("rad")              = rad;
   m.attr("radian")           = radian;
   m.attr("s")                = s;
   m.attr("second")           = second;
   m.attr("sr")               = sr;
   m.attr("steradian")        = steradian;
   m.attr("teraelectronvolt") = teraelectronvolt;
   m.attr("tesla")            = tesla;
   m.attr("TeV")              = TeV;
   m.attr("um")               = um;
   m.attr("us")               = us;
   m.attr("volt")             = volt;
   m.attr("watt")             = watt;
   m.attr("weber")            = weber;
}
