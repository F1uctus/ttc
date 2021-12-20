let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
  ] ++ (with pkgs.python3.pkgs; [
    black
    mypy
    spacy
    flake8
  ]);
}
