with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "DigitRecognition";
  env = buildEnv {
    name = name;
    paths = buildInputs;
  };
  buildInputs = with python36Packages; [
    gradle
    openjdk
  ];
}
