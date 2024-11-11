void parse_ttree_2particle() {
    const char* filename = "result_nuebar.root";
    const char* tree_name = "gtree_flat_2g";

    TFile *file = new TFile(filename, "READ");
    TTree *tree = (TTree*)file->Get(tree_name);
    
    TBranch *nu_energy = tree->GetBranch("init_nu_e");

    TBranch *p1em = tree->GetBranch("px1");
    TBranch *p2em = tree->GetBranch("py1");
    TBranch *p3em = tree->GetBranch("pz1");
    TBranch *p0em = tree->GetBranch("e1");

    TBranch *p1ep = tree->GetBranch("px2");
    TBranch *p2ep = tree->GetBranch("py2");
    TBranch *p3ep = tree->GetBranch("pz2");
    TBranch *p0ep = tree->GetBranch("e2");

    
    // Prepare the branch data variable and set the branch address
    float nu_e;

    float p1_electron;
    float p2_electron;
    float p3_electron;
    float p0_electron;
    
    float p1_positron;
    float p2_positron;
    float p3_positron;
    float p0_positron;

    nu_energy->SetAddress(&nu_e);
    p1em->SetAddress(&p1_electron);
    p2em->SetAddress(&p2_electron);
    p3em->SetAddress(&p3_electron);
    p0em->SetAddress(&p0_electron);
    p1ep->SetAddress(&p1_positron);
    p2ep->SetAddress(&p2_positron);
    p3ep->SetAddress(&p3_positron);
    p0ep->SetAddress(&p0_positron);


    std::ofstream flux;
    flux.open("2gamma/2gamma_nuebar_4vectors_DUNE_bkg.txt");

    // Loop over the entries and print the branch data for each entry
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        std::cout << "Entry " << entry << ": " << p1_electron << std::endl;
	flux << p0_electron << " " << p1_electron << " " \
		<< p2_electron << " " << p3_electron << " " << \
		p0_positron << " " << p1_positron << " " << \
		p2_positron << " " << p3_positron << " " << nu_e << std::endl;
    }

    flux.close();
    file->Close();
    delete file;
}

//int main() {
//    parse_ttree();
//    return 0;
//}
