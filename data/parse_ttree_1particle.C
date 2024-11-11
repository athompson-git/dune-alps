void parse_ttree_1particle() {
    const char* filename = "result_nuebar.root";
    const char* tree_name = "gtree_flat_1ep";

    TFile *file = new TFile(filename, "READ");
    TTree *tree = (TTree*)file->Get(tree_name);
    
    TBranch *nu_energy = tree->GetBranch("init_nu_e");

    TBranch *p1em = tree->GetBranch("px1");
    TBranch *p2em = tree->GetBranch("py1");
    TBranch *p3em = tree->GetBranch("pz1");
    TBranch *p0em = tree->GetBranch("e1");

    
    // Prepare the branch data variable and set the branch address
    float nu_e;

    float p1;
    float p2;
    float p3;
    float p0;

    nu_energy->SetAddress(&nu_e);
    p1em->SetAddress(&p1);
    p2em->SetAddress(&p2);
    p3em->SetAddress(&p3);
    p0em->SetAddress(&p0);


    std::ofstream flux;
    flux.open("1g0p/1ep_nuebar_4vectors_DUNE_bkg.txt");

    // Loop over the entries and print the branch data for each entry
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        std::cout << "Entry " << entry << ": " << p1 << std::endl;
	flux << p0 << " " << p1 << " " << p2 << " " << p3 << " " << nu_e << std::endl;
    }

    flux.close();
    file->Close();
    delete file;
}

//int main() {
//    parse_ttree();
//    return 0;
//}
