void parse_ttree() {
    const char* filename = "analysis_1em1ep_run_4_in_numubar_flux_numubar_1M_events.root";
    const char* tree_name = "gtree_flat";

    TFile *file = new TFile(filename, "READ");
    TTree *tree = (TTree*)file->Get(tree_name);
    
    TBranch *p1em = tree->GetBranch("pxem");
    TBranch *p2em = tree->GetBranch("pyem");
    TBranch *p3em = tree->GetBranch("pzem");
    TBranch *p0em = tree->GetBranch("eem");

    TBranch *p1ep = tree->GetBranch("pxep");
    TBranch *p2ep = tree->GetBranch("pyep");
    TBranch *p3ep = tree->GetBranch("pzep");
    TBranch *p0ep = tree->GetBranch("eep");

    
    // Prepare the branch data variable and set the branch address
    float p1_electron;
    float p2_electron;
    float p3_electron;
    float p0_electron;
    
    float p1_positron;
    float p2_positron;
    float p3_positron;
    float p0_positron;

    p1em->SetAddress(&p1_electron);
    p2em->SetAddress(&p2_electron);
    p3em->SetAddress(&p3_electron);
    p0em->SetAddress(&p0_electron);
    p1ep->SetAddress(&p1_positron);
    p2ep->SetAddress(&p2_positron);
    p3ep->SetAddress(&p3_positron);
    p0ep->SetAddress(&p0_positron);


    std::ofstream flux;
    flux.open("epem_numubar_4vectors_DUNE_bkg.txt");

    // Loop over the entries and print the branch data for each entry
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        std::cout << "Entry " << entry << ": " << p1_electron << std::endl;
	flux << p0_electron << " " << p1_electron << " " \
		<< p2_electron << " " << p3_electron << " " << \
		p0_positron << " " << p1_positron << " " << \
		p2_positron << " " << p3_positron << std::endl;
    }

    flux.close();
    file->Close();
    delete file;
}

//int main() {
//    parse_ttree();
//    return 0;
//}
