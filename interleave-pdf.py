# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "PyPDF2>=3.0.0",
# ]
# ///

import PyPDF2
import argparse

def interleave_pdfs(odd_file, even_file, output_file, reverse_even=True):
    """
    Interleaves pages from two PDFs into a single PDF.

    Parameters:
    odd_file (str): Path to the PDF containing odd-numbered pages (1, 3, 5...)
    even_file (str): Path to the PDF containing even-numbered pages (2, 4, 6...)
    output_file (str): Path to save the interleaved PDF
    reverse_even (bool): Whether to reverse the even pages (typically True when
                        you flip the stack and scan the back sides)
    """
    print(f"Interleaving '{odd_file}' and '{even_file}' into '{output_file}'...")

    # Open the PDFs
    with open(odd_file, 'rb') as odd_pdf_file, open(even_file, 'rb') as even_pdf_file:
        odd_pdf = PyPDF2.PdfReader(odd_pdf_file)
        even_pdf = PyPDF2.PdfReader(even_pdf_file)

        # Create a PDF writer object for the output file
        pdf_writer = PyPDF2.PdfWriter()

        # Get the number of pages in each PDF
        odd_pages = len(odd_pdf.pages)
        even_pages = len(even_pdf.pages)

        print(f"Odd PDF contains {odd_pages} pages")
        print(f"Even PDF contains {even_pages} pages")

        # Determine the total number of pages
        total_pages = odd_pages + even_pages

        # Prepare even pages (reverse if needed)
        even_pages_list = list(range(even_pages))
        if reverse_even:
            even_pages_list.reverse()

        # Interleave the pages
        for i in range(max(odd_pages, even_pages)):
            # Add odd page if available
            if i < odd_pages:
                pdf_writer.add_page(odd_pdf.pages[i])

            # Add even page if available
            if i < even_pages:
                pdf_writer.add_page(even_pdf.pages[even_pages_list[i]])

        # Write the interleaved PDF to the output file
        with open(output_file, 'wb') as output:
            pdf_writer.write(output)

    print(f"Successfully created interleaved PDF with {total_pages} pages.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interleave odd and even pages from two PDFs")
    parser.add_argument("odd_file", help="PDF file containing odd-numbered pages")
    parser.add_argument("even_file", help="PDF file containing even-numbered pages")
    parser.add_argument("output_file", help="Output PDF file name")
    parser.add_argument("--no-reverse", action="store_false", dest="reverse_even",
                        help="Don't reverse the even pages (use if even pages are already in correct order)")

    args = parser.parse_args()

    interleave_pdfs(args.odd_file, args.even_file, args.output_file, args.reverse_even)

