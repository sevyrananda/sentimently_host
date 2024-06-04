CREATE TABLE hasil_kuis_2 (
    id int(5) NOT NULL,
    total_skor_internal int(11) NOT NULL,
    total_skor_eksternal int(11) NOT NULL,
    hasil_kesimpulan text NOT NULL,
    create_by_ibu int(5) NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (create_by_ibu) REFERENCES data_ibu(id)
);